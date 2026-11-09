#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SinD Tianjin 8_2_1: onafhankelijke OD, roodlicht, wachten en spookrijden.

Geometrie en signaalparser komen uit het gecontroleerde v3-project. De v3
classifiers worden NIET gebruikt. Alle regels/drempels in deze module zijn
expliciet controleerbaar en worden in het validatierapport opgenomen.

Vanuit de SinD hoofdmap:
 python Tianjin_core_v4/analyse_tianjin_core_v4.py --plots
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from shapely.geometry import Point

from geometry_legacy_v3 import (ARMS, INWARD, DEFAULT_LIGHTS, METERS, Gate, Signals,
  Spatial, ensure_real_csv, read_map, read_tracks, role_class, movement_kind,
  kinematics, stopping_episodes, length_from_data, width_from_data,
  verify_tianjin_phases, unit)

# Explicit configuration; these are screening thresholds, NOT validated legal thresholds.
DEFAULTS = dict(endpoint_points=25, endpoint_min_support=0.55, endpoint_margin=0.20, endpoint_min_valid_points=3, endpoint_edge_points=6, endpoint_edge_required=2,
  endpoint_zone_reach_m=1.7, endpoint_min_upstream_m=1.0,
  crossing_max_gap_ms=700, crossing_min_forward_cos=0.55,
  crossing_hysteresis_m=0.05, crossing_context_ms=2000, phase_margin_ms=350,
  stop_speed_mps=0.40, stop_min_s=1.2, stop_spread_m=1.5,
  wrongway_speed_mps=0.7, wrongway_min_s=0.3, wrongway_min_m=2.0,
  wrongway_heading_cos=-0.60, wrongway_min_points=3,
  stop_position_tolerance_m=0.7, stop_distance_window_m=12.0,
  gate_extra_m=1.0, gate_full_width_extra_m=1.0)


def lanelet_stop_gates(roads):
    """Stopline mean station of final inbound approach_gap lanelets; separate physical
    inbound width and full road width (including outbound wrong-side approach).
    A gate is a transversal at a fixed station, not a curved painted marking.
    """
    physical={};virtual={};audit=[]
    for arm in ARMS:
        incoming=[z for z in roads if z.arm==arm and z.flow=='in']
        outgoing=[z for z in roads if z.arm==arm and z.flow=='out']
        gaps=[z for z in incoming if z.role=='approach_gap']
        ends=gaps if gaps else [z for z in incoming if z.role=='approach_or_departure']
        if not incoming or not outgoing or not ends:
            raise ValueError(f'Geen volledige in-/uitgaande Lanelet-geometry voor {arm}')
        norm=np.asarray(INWARD[arm],float)
        perp=np.array([-norm[1],norm[0]])
        # A physical lanelet's end is the user's stopline hypothesis.
        stations=np.array([np.dot(np.array(z.end),norm) for z in ends]);station=float(np.median(stations))
        if np.max(stations)-np.min(stations)>1.5:
            raise ValueError(f'{arm}: inkomende stopstreep-eindpunten liggen >1.5 m uit elkaar; geometrische inspectie vereist')
        pts=[]
        for z in incoming:
            # Calculate intersection with stopline (not just lanelet center endpoints).
            for xx,yy in z.poly.exterior.coords:
                v=np.asarray([xx,yy]);
                if abs(np.dot(v,norm)-station)<=4.0: pts.append(v)
        # In practice: use final gap polygons' transverse extent, excluding long road trapezoids.
        phys_ys=[float(np.dot(np.asarray([x,y]),perp)) for z in ends for x,y in z.poly.exterior.coords]
        if not phys_ys:raise ValueError('missing lanelet polygon vertices')
        center=station*norm
        def gate_from(tmin,tmax,kind):
            return Gate(arm, center+(tmin)*perp,center+(tmax)*perp,norm,True,kind,'user_assumed_lanelet_end_stopline')
        physical[arm]=gate_from(min(phys_ys)-DEFAULTS['gate_extra_m'],max(phys_ys)+DEFAULTS['gate_extra_m'],'physical_inbound_stopline_model')
        # Full width union covers both carriageway halves; use outgoing lane polygon
        # only close to intersection and project to the same transverse coordinate.
        nearby=incoming+outgoing
        full_ys=[float(np.dot(np.asarray([x,y]),perp)) for z in nearby for x,y in z.poly.exterior.coords
                 if abs(np.dot(np.asarray([x,y]),norm)-station)<=5.0]
        if not full_ys:raise ValueError(f'{arm}: no full corridor')
        virtual[arm]=gate_from(min(full_ys)-DEFAULTS['gate_full_width_extra_m'],max(full_ys)+DEFAULTS['gate_full_width_extra_m'],'virtual_full_carriageway_entry_line')
        for z in incoming:
            # The end of a long road is before the short gap; record whether lanelet
            # has a final gap alternative, and whether its lateral center is covered.
            y=float(np.dot(np.asarray(z.end),perp))
            audit.append(dict(arm=arm,lanelet_id=z.id,lanelet_name=z.name,lanelet_role=z.role,
                flow=z.flow,physical_covers_center=physical[arm].lateral_ok(z.end),
                virtual_covers_center=virtual[arm].lateral_ok(z.end),
                lanelet_end_station_m=round(float(np.dot(z.end,norm)),3),
                gate_station_m=round(station,3)))
        for z in outgoing:
            audit.append(dict(arm=arm,lanelet_id=z.id,lanelet_name=z.name,lanelet_role=z.role,
                flow=z.flow,physical_covers_center=physical[arm].lateral_ok(z.end),
                virtual_covers_center=virtual[arm].lateral_ok(z.end),
                lanelet_end_station_m=round(float(np.dot(z.end,norm)),3),gate_station_m=round(station,3)))
    return physical,virtual,pd.DataFrame(audit)


def endpoint_arm(g,spatial,gates,which,config):
    """Independent geographic road-arm inference; no demand for legal lane/flow.
    Points close to central intersection are not sufficient as origin/destination.
    """
    n=min(len(g),int(config['endpoint_points']))
    sub=g.head(n) if which=='start' else g.tail(n)
    votes=Counter();total=0;eligible=0;edge_votes=Counter()
    for index,row in enumerate(sub.itertuples(index=False)):
        p=Point(float(row.x),float(row.y))
        q=spatial.query(p,float(config['endpoint_zone_reach_m']))
        options={}
        for z,d,c in q:
            gate=gates.get(z.arm)
            if gate is None:continue
            station,_=gate.position((row.x,row.y))
            # station<0 upstream (outbound also upstream in space, despite its flow)
            if station > -float(config['endpoint_min_upstream_m']):continue
            score=float(d+.05*c)
            options[z.arm]=min(options.get(z.arm,1e9),score)
        total+=1
        if options:
            found=min(options,key=options.get)
            votes[found]+=1
            eligible+=1
            if (index<int(config['endpoint_edge_points']) if which=='start' else index>=n-int(config['endpoint_edge_points'])):
                edge_votes[found]+=1
    if not votes:return '',0.0,'no_geographic_road_zone'
    arm,k=votes.most_common(1)[0]
    v2=votes.most_common(2)[1][1] if len(votes)>1 else 0
    # Vote among actually observed road-zone points, not empty intersection samples.
    # Require >=3 valid samples to avoid classifying on a single accidental hit.
    support=k/max(eligible,1)
    if (eligible<int(config['endpoint_min_valid_points'])
        or edge_votes[arm]<int(config['endpoint_edge_required'])
        or support<float(config['endpoint_min_support'])
        or (k-v2)/max(eligible,1)<float(config['endpoint_margin'])):
        return '',round(support,3),'ambiguous_or_incomplete_endpoint'
    return arm,round(support,3),'geographic_road_arm_zone'


def movement_support(g,origin,dest):
    """Do not demand legal lane use. Check gross movement toward/from intersection.
    Do not overwrite OD on failure; add a caution status instead.
    """
    if not origin or not dest:return 'not_assessable'
    if len(g)<5:return 'too_few_samples'
    xy=g[['x','y']].to_numpy(float)
    n=min(12,max(2,len(g)//5))
    ds=xy[n-1]-xy[0];de=xy[-1]-xy[-n]
    startcos=float(np.dot(unit(ds),np.asarray(INWARD[origin]))) if unit(ds) is not None else 0
    endcos=float(np.dot(unit(de),-np.asarray(INWARD[dest]))) if unit(de) is not None else 0
    if startcos>0.35 and endcos>0.35:return 'direction_consistent'
    if startcos<=0.35 and endcos>0.35:return 'start_direction_unclear'
    if startcos>0.35 and endcos<=0.35:return 'end_direction_unclear'
    return 'both_directions_unclear'


def robust_front(g,length_m,origin,vel):
    """Front reference: vehicle center + 0.5 length along measured travel direction.
    If orientation absent/uncertain no silent center-as-front; indicate proxy.
    No arbitrary per-frame min motion threshold.
    """
    xy=g[['x','y']].to_numpy(float);n=len(xy)
    forward=np.full(n,np.nan)
    if not origin:return xy,forward,'unknown_origin_reference_only'
    inward=np.asarray(INWARD[origin],float)
    for i in range(n):
        v=vel[i]
        if np.all(np.isfinite(v)) and np.linalg.norm(v)>.15:
            forward[i]=float(np.dot(v/np.linalg.norm(v),inward))
    if length_m is None:return xy,forward,'track_reference_no_measured_length'
    # Use underlying track heading if measured, rather than moving heading where
    # noisy at slow speeds. For front estimate during stop use nearby valid v.
    nearest=np.where(np.isfinite(forward))[0]
    f=forward.copy()
    for i in range(n):
        if not np.isfinite(f[i]) and len(nearest):
            j=nearest[np.argmin(np.abs(nearest-i))]
            if abs(float(g.timestamp_ms.iloc[i]-g.timestamp_ms.iloc[j]))<=3000:f[i]=f[j]
    front=xy.copy()
    for i in range(n):
        if np.isfinite(f[i]) and f[i]>.3:
            # Project front on inward axis. Crossing calculations need projected
            # front; no perpendicular approximation for curving approach.
            front[i]+=float(length_m)*.5*f[i]*inward
    return front,f,'estimated_front_from_length_and_velocity'


def crossings(g,gate,virtual,signals,sid,origin,destination,vel,length_m,config,usable_signal=True):
    """Detect entrance crossing of center/front. Separately track physical vs
    virtual-only line. Crossing direction is intersection-inbound; no reverse exits.
    Hysteresis requires at least one upstream sample, but no min movement size.
    """
    if not origin or gate is None or virtual is None:return []
    xy=g[['x','y']].to_numpy(float);t=g.timestamp_ms.to_numpy(float)
    front,fwd,reference=robust_front(g,length_m,origin,vel)
    d=np.array([virtual.position(p)[0] for p in front]);events=[]
    margin=float(config['crossing_hysteresis_m'])
    normal=np.asarray(INWARD[origin],float)
    for i in range(1,len(g)):
        dt=t[i]-t[i-1]
        if not (0<dt<=float(config['crossing_max_gap_ms'])):continue
        dx=xy[i]-xy[i-1]; u=unit(dx)
        # Small moves are allowed; for stationary jitter, require convincing
        # approach direction from the smoothed velocity (not just dx).
        v=vel[i] if np.all(np.isfinite(vel[i])) else vel[i-1]
        vv=unit(v) if np.all(np.isfinite(v)) else None
        aligned=float(np.dot(vv,normal)) if vv is not None else float(np.dot(u,normal)) if u is not None else 0.
        if aligned<float(config['crossing_min_forward_cos']):continue
        if not (d[i-1]<-margin and d[i]>=-margin) and not (d[i-1]<0<=d[i]):continue
        if d[i]<=d[i-1]:continue
        # Actual geometric zero-crossing. Require evidence of sustained positions
        # on BOTH sides over up to 2 seconds, without enforcing a minimum
        # displacement in one sample. Reject jitter around the threshold.
        if not (d[i-1]<0<=d[i]):continue
        lookback=int(np.searchsorted(t,t[i]-float(config['crossing_context_ms']),'left'))
        lookahead=int(np.searchsorted(t,t[i]+float(config['crossing_context_ms']),'right'))
        if np.min(d[lookback:i+1])>-margin or np.max(d[i:lookahead])<margin:continue
        f=float(-d[i-1]/(d[i]-d[i-1]));cp=front[i-1]+f*(front[i]-front[i-1]);tc=float(t[i-1]+f*dt)
        if not virtual.lateral_ok(cp):continue
        physical=bool(gate.lateral_ok(cp))
        state,stable=signals.at(sid,tc) if sid is not None and usable_signal else ('unknown',False)
        move=movement_kind(origin,destination)
        if state=='red' and move=='right':typ='right_turn_on_red';verdict='exempt_right_turn_on_red'
        elif state=='red' and move in ('straight','left'):
            typ='red_light_running_candidate' if stable else 'red_crossing_near_phase_transition'
            verdict='suspected_violation' if stable else 'uncertain_phase_transition'
        elif state=='red':typ='red_crossing_movement_unknown';verdict='not_assessable_movement'
        elif state=='green':typ='green_crossing';verdict='control_not_violation'
        elif state=='yellow':typ='yellow_crossing';verdict='yellow_not_automatically_violation'
        else:typ='crossing_signal_unknown';verdict='not_assessable_signal'
        events.append(dict(event_type=typ,detector='signal',track_id=str(g.track_id.iloc[0]),origin=origin,
            destination=destination,movement=move,start_ms=round(tc,2),end_ms=round(tc,2),
            x_m=round(float(cp[0]),3),y_m=round(float(cp[1]),3),signal_id=sid or '',signal_state=state,
            phase_stable=bool(stable),gate_kind='physical_lanelet_stopline' if physical else 'virtual_extension_wrong_side_or_outer_lane',
            reference_point=reference,interpretation=verdict,
            evidence='front_edge_crossed_from_upstream_to_downstream',length_m=length_m if length_m else '',
            physical_line_covered=physical))
    # One approach, one first entrance crossing; later crossings can occur during
    # peculiar loops, retain as separate events only when sufficiently separated.
    return [e for j,e in enumerate(events) if j==0 or e['start_ms']-events[j-1]['start_ms']>2000]


def wait_events(g,gate,virtual,signals,sid,origin,destination,vel,speed,length_m,config,crossing_events):
    if not origin or virtual is None:return []
    stops=stopping_episodes(g,vel,speed,float(config['stop_speed_mps']),
                            float(config['stop_min_s']),float(config['stop_spread_m']))
    xy=g[['x','y']].to_numpy(float);front,fw,ref=robust_front(g,length_m,origin,vel)
    out=[]
    for ids in stops:
        t0=float(g.timestamp_ms.iloc[ids[0]]);t1=float(g.timestamp_ms.iloc[ids[-1]])
        red=signals.during(sid,t0,t1,'red') if sid is not None else 0
        if red<0.5:continue
        p=np.median(front[ids],axis=0)
        if not virtual.lateral_ok(p):continue
        d=float(virtual.position(p)[0]);tol=float(config['stop_position_tolerance_m'])
        if abs(d)>float(config['stop_distance_window_m']):continue
        if d>tol:where='past_stopline'
        elif d< -tol:where='before_stopline'
        else:where='on_or_partly_over_stopline'
        prev=[e for e in crossing_events if e['start_ms']<=t1]
        if prev:history='observed_'+str(prev[-1]['signal_state'])+'_crossing'
        elif d>tol:history='crossing_not_observed_left_censored_or_untracked'
        else:history='no_crossing_observed'
        out.append(dict(event_type='wait_during_red',detector='waiting',track_id=str(g.track_id.iloc[0]),
            origin=origin,destination=destination,movement=movement_kind(origin,destination),
            start_ms=round(t0,2),end_ms=round(t1,2),x_m=round(float(p[0]),3),y_m=round(float(p[1]),3),
            signal_id=sid or '',signal_state='red',phase_stable='',
            gate_kind='physical_lanelet_stopline' if gate and gate.lateral_ok(p) else 'virtual_extension',
            reference_point=ref,interpretation='observed_waiting_not_independent_violation',
            evidence=where+';'+history,length_m=length_m if length_m else '',
            physical_line_covered=bool(gate and gate.lateral_ok(p)),stop_position=where,
            crossing_history=history,red_overlap_s=round(red,2)))
    return out


def wrong_way_events(g,road_spatial,vel,speed,config):
    """Only physical straight arm lanes. Points in connector/intersection excluded.
    Opposite spatial half is not alone enough: sustained anti-lane motion required.
    """
    xy=g[['x','y']].to_numpy(float);t=g.timestamp_ms.to_numpy(float)
    mask=[];chosen=[];reasons=[]
    for i,p in enumerate(xy):
        pt=Point(*p)
        cand=[(z,d,c) for z,d,c in road_spatial.query(pt,.20)
              if z.role in ('approach_or_departure','approach_gap') and z.arm in ARMS]
        if not cand or not np.isfinite(speed[i]) or speed[i]<float(config['wrongway_speed_mps']):
            mask.append(False);chosen.append(None);reasons.append('not_on_road_or_slow');continue
        # Choose nearest polygon centerline; avoid claiming if equally plausible
        # lanes with incompatible permitted flow overlap.
        cand.sort(key=lambda x:(x[1],x[2]))
        b=cand[0];z=b[0]
        amb=[a for a in cand if abs(a[1]-b[1])<.06 and abs(a[2]-b[2])<.4 and a[0].flow!=z.flow]
        if amb:mask.append(False);chosen.append(None);reasons.append('ambiguous_opposite_lane_overlap');continue
        allowed=np.asarray(INWARD[z.arm],float)*(1 if z.flow=='in' else -1)
        u=unit(vel[i]);cos=float(np.dot(u,allowed)) if u is not None else 0.
        yes=cos<=float(config['wrongway_heading_cos'])
        mask.append(yes);chosen.append(z if yes else None);reasons.append('against' if yes else 'not_against')
    # Group only sustained wrong-way observations; allow small gaps of lost lane
    # matching but never count mere transverse crossing as a violation.
    ev=[];groups=[];current=[]
    for i,yes in enumerate(mask):
        if yes:
            if current and (t[i]-t[current[-1]]>500 or chosen[i].arm!=chosen[current[-1]].arm):
                groups.append(current);current=[]
            current.append(i)
        elif current:groups.append(current);current=[]
    if current:groups.append(current)
    for ids in groups:
        distance=float(np.sum(np.linalg.norm(np.diff(xy[ids],axis=0),axis=1))) if len(ids)>1 else 0.
        duration=(t[ids[-1]]-t[ids[0]])/1000
        if len(ids)<int(config['wrongway_min_points']) or distance<float(config['wrongway_min_m']) or duration<float(config['wrongway_min_s']):continue
        zids=sorted({chosen[i].id for i in ids})
        pt=np.mean(xy[ids],axis=0)
        ev.append(dict(event_type='wrong_way_candidate',detector='wrong_way',track_id=str(g.track_id.iloc[0]),
            origin='',destination='',movement='',start_ms=round(float(t[ids[0]]),2),end_ms=round(float(t[ids[-1]]),2),
            x_m=round(float(pt[0]),3),y_m=round(float(pt[1]),3),signal_id='',signal_state='',phase_stable='',
            gate_kind='',reference_point='tracked_position',interpretation='suspected_violation',
            evidence='sustained_motion_opposite_to_lane_direction;'+','.join(zids),
            length_m='',physical_line_covered='',travel_distance_m=round(distance,3),duration_s=round(duration,3),
            lanelet_ids=';'.join(zids),arm=chosen[ids[0]].arm))
    return ev


def summarize(tid,g,atype,origin,dest,od_status,support1,support2,headingstatus,ev,signal_ok):
    red=[e for e in ev if e['event_type']=='red_light_running_candidate']
    wrong=[e for e in ev if e['event_type']=='wrong_way_candidate']
    right=[e for e in ev if e['event_type']=='right_turn_on_red']
    boundary=[e for e in ev if e['event_type']=='red_crossing_near_phase_transition']
    stops=[e for e in ev if e['event_type']=='wait_during_red']
    sig=[e for e in ev if e['detector']=='signal']
    if not origin or not dest or headingstatus!='direction_consistent':
        od_status='partial_or_ambiguous' if not (origin and dest) else 'classified_with_direction_caution'
    signal_status='not_assessable_no_origin' if not origin else ('not_assessable_crossing_not_observed' if not sig else
      'suspected_red_light_violation' if red else 'transition_red_review_required' if boundary else 'right_turn_on_red_exempt' if right else
      'crossing_observed_no_illegal_red_detected')
    wrongstatus='wrong_way_detected' if wrong else 'no_wrong_way_detected_on_assessable_segments'
    if not ev:overall='no_deviation_detected_not_proof_of_compliance'
    elif red or wrong:overall='suspected_violation'
    elif boundary:overall='phase_transition_manual_review'
    else:overall='observed_behavior_not_independently_violation'
    return dict(track_id=tid,agent_type=atype,n_points=len(g),start_ms=float(g.timestamp_ms.iloc[0]),end_ms=float(g.timestamp_ms.iloc[-1]),
      origin=origin,destination=dest,movement=movement_kind(origin,dest),od_status=od_status,
      origin_support=support1,destination_support=support2,endpoint_direction_consistency=headingstatus,
      signal_status=signal_status,wrong_way_status=wrongstatus,
      wait_before_count=sum(e.get('stop_position')=='before_stopline' for e in stops),
      wait_on_count=sum(e.get('stop_position')=='on_or_partly_over_stopline' for e in stops),
      wait_past_count=sum(e.get('stop_position')=='past_stopline' for e in stops),
      red_violation_candidates=len(red),red_transition_review_count=len(boundary),right_turn_on_red_count=len(right),wrong_way_candidates=len(wrong),
      n_signal_crossings=len(sig),n_wait_episodes=len(stops),overall_classification=overall,
      signal_phase_timeline_checked=bool(signal_ok))


def make_plot(g,phys,virt,events,origin,dest,out):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,7));xy=g[['x','y']].to_numpy(float)
    ax.plot(xy[:,0],xy[:,1],lw=1.6,label='trajectory')
    ax.scatter(*xy[0],s=35,marker='o',label='first');ax.scatter(*xy[-1],s=35,marker='s',label='last')
    for arm in ARMS:
        for gate,style,label in ((phys[arm],'-','physical stopline'),(virt[arm],'--','extended virtual line')):
            ax.plot([gate.a[0],gate.b[0]],[gate.a[1],gate.b[1]],ls=style,lw=1.0,label=label if arm=='N' else None)
    for ev in events:
        ax.scatter([ev['x_m']],[ev['y_m']],marker='x',s=70,label=ev['event_type'])
    ax.set_aspect('equal');ax.grid(alpha=.25)
    ax.set_title(f"Vehicle {str(g.track_id.iloc[0])}: {origin or '?'} -> {dest or '?'}")
    ax.legend(fontsize=7,loc='best');fig.tight_layout();fig.savefig(out,dpi=140);plt.close(fig)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tracks',type=Path,default=Path('Data/Tianjin/8_2_1/Veh_smoothed_tracks.csv'))
    ap.add_argument('--meta',type=Path,default=Path('Data/Tianjin/8_2_1/Veh_tracks_meta.csv'))
    ap.add_argument('--signals',type=Path,default=Path(__file__).with_name('TrafficLight_8_2_1.csv'))
    ap.add_argument('--osm',type=Path,default=Path(__file__).with_name('tianjin_lanelets_uitgebreid.osm'))
    ap.add_argument('--output-dir',type=Path,default=Path('analysis_outputs/lanelet_audit/tianjin_core_v4'))
    ap.add_argument('--plots',action='store_true')
    ap.add_argument('--plot-ids',default='1,2,4,9,10,11,12,17')
    ap.add_argument('--config',type=Path)
    ap.add_argument('--allow-missing-meta-for-diagnostics',action='store_true',help='Alleen diagnostische audit zonder echte metadata')
    args=ap.parse_args(argv)
    cfg=dict(DEFAULTS)
    if args.config:
        overrides=json.loads(args.config.read_text(encoding='utf-8'))
        if not set(overrides).issubset(cfg):raise ValueError('Onbekende config keys: '+str(set(overrides)-set(cfg)))
        cfg.update(overrides)
    ensure_real_csv(args.tracks);ensure_real_csv(args.signals)
    if args.meta.is_file():ensure_real_csv(args.meta)
    elif not args.allow_missing_meta_for_diagnostics:
        raise FileNotFoundError('Veh_tracks_meta.csv ontbreekt: zonder echte metadata geen volledige validatie. '+
                                'Voor uitsluitend diagnostische puntbestanden gebruik --allow-missing-meta-for-diagnostics.')
    lanes,dropped=read_map(args.osm)
    roads=[z for z in lanes if z.role in ('approach_or_departure','approach_gap')]
    road_spatial=Spatial(roads)
    physical,virtual,coverage=lanelet_stop_gates(roads)
    signals=Signals(args.signals,transition_margin_ms=cfg['phase_margin_ms'])
    check=verify_tianjin_phases(signals,strict=True)
    if not check['phase_group_verified']:raise ValueError('Verkeerslichtfasegroepen inconsistent')
    tracks,meta=read_tracks(args.tracks,args.meta)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    coverage.to_csv(out/'stoplijn_coverage.csv',index=False)
    (out/'config_actief.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    vehicle_rows=[];all_events=[];plots={str(x.strip()) for x in args.plot_ids.split(',')}
    for tid,g in tracks.groupby('track_id',sort=False):
        g=g.reset_index(drop=True)
        md=meta.get(tid,{})
        atype=str(g.agent_type.iloc[0]).strip() if 'agent_type' in g else str(md.get('class','unknown'))
        if atype.lower() in ('','nan','none'):atype=str(md.get('class','unknown'))
        origin,cs,os=endpoint_arm(g,road_spatial,virtual,'start',cfg)
        dest,cd,ds=endpoint_arm(g,road_spatial,virtual,'end',cfg)
        headingstatus=movement_support(g,origin,dest)
        vel,speed=kinematics(g)
        length,ls=length_from_data(g,md)
        # Research assumption: nonmotor uses same N/S or E/W motor signal as cars.
        sid=DEFAULT_LIGHTS.get(origin)
        cross=crossings(g,physical.get(origin),virtual.get(origin),signals,sid,origin,dest,vel,length,cfg)
        stops=wait_events(g,physical.get(origin),virtual.get(origin),signals,sid,origin,dest,vel,speed,length,cfg,cross)
        wrong=wrong_way_events(g,road_spatial,vel,speed,cfg)
        ev=cross+stops+wrong
        for e in ev:
            e['agent_type']=atype
            if not e['origin']:e['origin']=origin
            if not e['destination']:e['destination']=dest
            if not e['movement']:e['movement']=movement_kind(origin,dest)
        all_events.extend(ev)
        row=summarize(tid,g,atype,origin,dest,'classified' if origin and dest else 'incomplete',cs,cd,headingstatus,ev,check['timeline_verified'])
        row.update(meta_Signal_Violation_Behavior=md.get('Signal_Violation_Behavior',''),
                   vehicle_length_m=length if length else '',length_source=ls,
                   origin_zone_reason=os,destination_zone_reason=ds)
        vehicle_rows.append(row)
        if args.plots and tid in plots:
            (out/'plots').mkdir(exist_ok=True)
            make_plot(g,physical,virtual,ev,origin,dest,out/'plots'/f'track_{tid}.png')
    vehicles=pd.DataFrame(vehicle_rows);evdf=pd.DataFrame(all_events)
    vehicles.to_csv(out/'per_voertuig.csv',index=False,encoding='utf-8-sig')
    evdf.to_csv(out/'gebeurtenissen.csv',index=False,encoding='utf-8-sig')
    gates=[]
    for arm in ARMS:
        for typ,gg in [('physical',physical[arm]),('virtual',virtual[arm])]:
            gates.append(dict(arm=arm,type=typ,x1=gg.a[0],y1=gg.a[1],x2=gg.b[0],y2=gg.b[1]))
    pd.DataFrame(gates).to_csv(out/'stoplijnen_en_virtuele_lijnen.csv',index=False)
    # Validations: metadata not used in detection. These are not confusion-matrix metrics.
    label=vehicles['meta_Signal_Violation_Behavior'].fillna('').astype(str)
    review=pd.crosstab(label,vehicles['red_violation_candidates']>0,dropna=False)
    review.to_csv(out/'vergelijking_met_SinD_labels.csv')
    gates_ok=coverage.groupby('arm')['virtual_covers_center'].all().to_dict()
    # Annotated disagreements are audit targets, not accuracy estimates.
    annotated=vehicles['meta_Signal_Violation_Behavior'].fillna('').astype(str)
    review_cases=vehicles.loc[((annotated=='red-light running') & (vehicles['red_violation_candidates']==0)) |
        ((annotated=='No violation of traffic lights') & (vehicles['red_violation_candidates']>0)),
        ['track_id','agent_type','origin','destination','movement','signal_status',
         'red_violation_candidates','red_transition_review_count','meta_Signal_Violation_Behavior']].copy()
    review_cases.to_csv(out/'prioritaire_discrepanties.csv',index=False)
    report=dict(version='v4',traffic_light_groups_checked=check,number_tracks=len(vehicles),
        number_points=len(tracks),events=evdf.event_type.value_counts().to_dict() if len(evdf) else {},
        metadata_discrepancies_to_review=len(review_cases),
        incomplete_od=int(vehicles.origin.eq('').sum()+vehicles.destination.eq('').sum()),
        stopline_coverage_by_arm={k:bool(v) for k,v in gates_ok.items()},
        lanelets_unreadable=dropped,assumptions=[
           'lanelet approach_gap endpoints represent stopline',
           'nonmotor follows motor approach phase (user research assumption)',
           'right turns on red not counted as red violations; yielding excluded',
           'only observed trajectories are assessed; absence of detection is not proof of compliance'],
        limitations=['No original full trajectories in sandbox if using reconstructed v3 point export',
           'No independent lane geometry or manual labels for the full dataset',
           'Measured length may not specify reference point/heading semantics',
           'Straight approach lane against-direction is screened; not every possible wrong-way maneuver'])
    report['metadata_file_present']=bool(args.meta.is_file())
    report['input_file_name']=str(args.tracks)
    (out/'validatierapport.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    checks=[
      {'check':'phase_groups_and_replay_timeline','status':'PASS' if check['timeline_verified'] else 'FAIL', 'detail':str(check)},
      {'check':'all_arm_virtual_line_geometry','status':'PASS' if all(gates_ok.values()) else 'FAIL', 'detail':str(gates_ok)},
      {'check':'genuine_metadata_available','status':'PASS' if args.meta.is_file() else 'NOT_TESTED', 'detail':str(args.meta)},
      {'check':'red_light_metadata_comparison','status':'COMPLETED' if args.meta.is_file() else 'NOT_TESTED', 'detail':f'{len(review_cases)} candidates for independent review'},
      {'check':'real_world_detection_accuracy','status':'NOT_VALIDATED', 'detail':'Requires independently checked video ground truth, not synthetic tests or metadata alone'},
      {'check':'all_physical_lanes_stopline_alignment','status':'ASSUMED_NOT_MEASURED','detail':'User hypothesis: incoming approach_gap ends equal stoplines'},
      {'check':'separate_bicycle_signal_applicability','status':'ASSUMED', 'detail':'User assumption: nonmotorized follow main approach phase'},
    ]
    pd.DataFrame(checks).to_csv(out/'validatierapport.csv',index=False)
    print('Verwerkt:',len(vehicles),'voertuigen,',len(tracks),'punten')
    print('OD:',vehicles.od_status.value_counts().to_dict())
    print('Gebeurtenissen:',report['events'])
    print('Signaalcontrole:',check['timeline_verified'],'Stoplijncoverage:',gates_ok)
    print('Uitvoer:',out.resolve())
    return 0

if __name__=='__main__':sys.exit(main())
