#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tianjin SinD - OD, Lanelet2, signalen, stoppen en bochtgedrag: controleerbaar prototype.

Alle tags van de uitgebreide kaart worden als hypotheses behandeld; geen juridische
schuldigverklaring. Werkt zelfstandig (geen imports van eerdere ITS-scripts).

Vanuit SinD-root:
    python analyse_tianjin_v3.py --plots --plot-ids 1,2,4,9,10,11,12,17
Opties: --stoplines stoplijnen_gevalideerd.csv --signal-config signaalregels.json
         --signal-offset-ms 0 --tracks ... --meta ... --signals ... --osm ...
Vereist: pip install numpy pandas shapely matplotlib
"""
from __future__ import annotations
import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
from shapely.strtree import STRtree

METERS = 111319.49079327358  # Exact dezelfde projectie als het vorige Lanelet-script
ARMS = ('N', 'E', 'S', 'W')
INWARD = {'N': (0., -1.), 'E': (-1., 0.), 'S': (0., 1.), 'W': (1., 0.)}
DEFAULT_LIGHTS = {'N': 1, 'E': 7, 'S': 5, 'W': 3}
LIGHT_CODES = {0: 'red', 1: 'green', 3: 'yellow'}
ROAD_NAME = re.compile(r'R([1-4])(In|Out)([1-3])$', re.I)
R_ARMS = {'1': 'E', '2': 'N', '3': 'W', '4': 'S'}
CON_NAME = re.compile(r'([NESW])_EX_(\d+)_to_([NESW])_EN_(\d+)$', re.I)
MOTOR = {'car', 'motorcycle', 'truck', 'bus', 'van', 'taxi', 'suv'}
NONMOTOR = {'bicycle', 'bike', 'cyclist', 'electric_bicycle', 'e-bicycle', 'ebike'}
EVENT_COLUMNS = ['track_id','agent_type','vehicle_category','origin','destination','movement',
                 'event_type','evidence_level','review_status','reason','zone','arm',
                 'start_ms','end_ms','duration_s','samples','travel_distance_m',
                 'lanelet_ids','x_m','y_m','signal_id','signal_state','signal_applicability',
                 'signal_mapping_verified','time_alignment_verified','phase_stable',
                 'line_type','line_validated','reference_point','detail','duplicate_of','assessment']
POINT_COLUMNS = ['track_id','agent_type','timestamp_ms','x','y','speed_mps','lanelet_id',
                 'lanelet_name','arm','flow','lane_role','direction_cosine','motion',
                 'assignment_status','centerline_distance_m','on_crosswalk','crosswalk_id']


def tags(el):
    return {t.get('k'): t.get('v') for t in el.findall('tag')}


def ensure_real_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'Ontbrekend bestand: {path}')
    with path.open('rb') as f:
        if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'{path} is een Git-LFS-pointer, niet de dataset. Gebruik git lfs pull in je repo.')


def clean_id(v):
    s = str(v).strip()
    return s[:-2] if s.endswith('.0') else s


def numeric(v, default=None):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def unit(v):
    a = np.asarray(v, float)
    n = np.linalg.norm(a)
    return a / n if n > 1e-9 else None


def role_class(agent):
    a = str(agent).lower().replace(' ', '_').strip()
    if a in MOTOR: return 'motorized'
    if a in NONMOTOR: return 'non_motorized'
    # SinD heeft 'tricycle'; elektrisch/motorisch onbekend -> NIET automatisch fiets.
    return 'uncertain'


@dataclass
class Lane:
    id: str
    name: str
    role: str
    arm: str
    flow: str
    lane_role: str
    from_arm: str
    to_arm: str
    movement: str
    marked_movements: str
    direction_status: str
    poly: Polygon
    center: LineString
    start: tuple
    end: tuple
    bbox: tuple


def read_map(path):
    root = ET.parse(path).getroot()
    nodes = {e.get('id'): (float(e.get('lon'))*METERS, float(e.get('lat'))*METERS)
             for e in root.findall('node')}
    ways = {}
    for w in root.findall('way'):
        ways[w.get('id')] = [nodes[nd.get('ref')] for nd in w.findall('nd') if nd.get('ref') in nodes]
    lanes, dropped = [], []
    for rel in root.findall('relation'):
        t = tags(rel)
        if t.get('type') != 'lanelet': continue
        lid = rel.get('id')
        mem = {x.get('role'): x.get('ref') for x in rel.findall('member') if x.get('type') == 'way'}
        a, b = ways.get(mem.get('left'), []), ways.get(mem.get('right'), [])
        if len(a) < 2 or len(b) < 2:
            dropped.append((lid, 'missing_boundary')); continue
        if math.dist(a[0], b[-1]) + math.dist(a[-1], b[0]) < math.dist(a[0], b[0]) + math.dist(a[-1], b[-1]):
            b = b[::-1]
        poly = Polygon(a + b[::-1])
        if not poly.is_valid: poly = poly.buffer(0)
        if poly.is_empty or poly.area < 0.02:
            dropped.append((lid, 'invalid_polygon')); continue
        aa, bb = LineString(a), LineString(b)
        centers = []
        for s in np.linspace(0, 1, 25):
            p = aa.interpolate(float(s), normalized=True)
            q = bb.interpolate(float(s), normalized=True)
            centers.append(((p.x+q.x)/2, (p.y+q.y)/2))
        center = LineString(centers)
        name = t.get('name', '')
        role, arm, flow = t.get('its:role',''), t.get('its:arm',''), t.get('its:flow','')
        origin, dest = t.get('its:from_arm',''), t.get('its:to_arm','')
        if not role:
            m = ROAD_NAME.fullmatch(name)
            if m:
                role, arm, flow = 'approach_or_departure', R_ARMS[m.group(1)], m.group(2).lower()
            elif CON_NAME.fullmatch(name):
                m = CON_NAME.fullmatch(name)
                role, origin, dest = 'lane_specific_connector', m.group(1), m.group(3)
            elif t.get('subtype') == 'crosswalk': role = 'crosswalk'
            else: role = 'other'
        status = t.get('its:direction_audit', '')
        if role in ('approach_or_departure','approach_gap'):
            expected = np.asarray(INWARD[arm]) * (1 if flow == 'in' else -1)
            v = np.asarray(centers[-1]) - np.asarray(centers[0])
            if unit(v) is None or float(np.dot(unit(v), expected)) < .65: status = 'unverified_or_inconsistent'
            else: status = 'geometry_consistent_direction_assumed'
        elif status == 'reversed_geometry': status = 'reversed_geometry_do_not_use_for_direction'
        elif role == 'lane_specific_connector' and status == 'consistent': status = 'geometry_name_consistent_direction_assumed'
        else: status = 'orientation_unverified'
        lanes.append(Lane(lid, name, role, arm, flow, t.get('its:lane_role','unknown'),
                          origin, dest, t.get('its:movement',''), t.get('its:marked_movements',''),
                          status, poly, center, centers[0], centers[-1], poly.bounds))
    return lanes, dropped


class Spatial:
    def __init__(self, lanes):
        self.lanes = list(lanes)
        self.geoms = [z.poly for z in self.lanes]
        self.tree = STRtree(self.geoms) if self.geoms else None
        self.index = {id(g):i for i,g in enumerate(self.geoms)}
    def query(self,p,reach=.5):
        if self.tree is None:return []
        matches = []
        for obj in self.tree.query(p if reach <= 0 else p.buffer(reach)):
            i = int(obj) if isinstance(obj,(int,np.integer)) else self.index.get(id(obj),-1)
            if i < 0: continue
            z = self.lanes[i]
            d = z.poly.distance(p)
            if d <= reach: matches.append((z,d,z.center.distance(p)))
        return matches


def read_tracks(path, meta):
    ensure_real_csv(path)
    data = pd.read_csv(path, low_memory=False)
    need = {'track_id','timestamp_ms','x','y'}
    if not need.issubset(data):raise ValueError(f'Trajectdata missen {need-set(data.columns)}')
    for c in ['timestamp_ms','x','y']:data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna(subset=['track_id','timestamp_ms','x','y']).copy()
    data['track_id'] = data['track_id'].map(clean_id)
    if 'agent_type' not in data.columns:data['agent_type'] = ''
    data = data.sort_values(['track_id','timestamp_ms']).drop_duplicates(['track_id','timestamp_ms']).reset_index(drop=True)
    if data.empty:raise ValueError('Geen trajectpunten gevonden')
    if meta and meta.is_file():
        ensure_real_csv(meta)
        d = pd.read_csv(meta,low_memory=False)
        if 'trackId' in d.columns:
            d['trackId'] = d['trackId'].map(clean_id)
            lookup = d.drop_duplicates('trackId').set_index('trackId').to_dict('index')
        else:lookup={}
    else:lookup={}
    return data,lookup


def kinematics(g,window_ms=500):
    """Puntsnelheden uit symmetrische tijdvensters; geen bridging over gaten > 0.8s."""
    n=len(g); x=g.x.to_numpy(float); y=g.y.to_numpy(float);t=g.timestamp_ms.to_numpy(float)
    vx=np.full(n,np.nan);vy=np.full(n,np.nan);speeds=np.full(n,np.nan)
    for i in range(n):
        lo = max(0,int(np.searchsorted(t,t[i]-window_ms/2,side='left')))
        hi = min(n-1,int(np.searchsorted(t,t[i]+window_ms/2,side='right')-1))
        if hi<=lo:continue
        dt=(t[hi]-t[lo])/1000
        if dt<.08 or dt>.75 or np.max(np.diff(t[lo:hi+1]))>750:continue
        vx[i] = (x[hi]-x[lo])/dt
        vy[i] = (y[hi]-y[lo])/dt
        speeds[i] = math.hypot(vx[i],vy[i])
    return np.column_stack((vx,vy)),speeds


def infer_arm(g,road_spatial,side,reach=1.7):
    n=min(len(g),25)
    sub = g.head(n) if side=='origin' else g.tail(n)
    votes=Counter()
    candidates=[]
    for row in sub.itertuples(index=False):
        p=Point(float(row.x),float(row.y))
        qs=road_spatial.query(p,reach)
        if not qs:continue
        # Neem arm-geometrie, niet flow: tegenrichting mag OD niet automatisch veranderen.
        by_arm={}
        for z,d,c in qs:
            val = d+0.05*c
            if z.arm not in by_arm or val<by_arm[z.arm]:by_arm[z.arm]=val
        arm=min(by_arm,key=by_arm.get)
        votes[arm]+=1
        candidates.append(arm)
    if not votes:return ('',0.0,'no_road_arm_near_endpoint')
    arm,nvotes=votes.most_common(1)[0]
    share=nvotes/n
    second=votes.most_common(2)[1][1]/n if len(votes)>1 else 0
    if share<.45 or share-second<.2:
        return ('',round(share,3),'ambiguous_endpoint')
    return arm,round(share,3),'geographic_corridor'


def movement_kind(origin,dest):
    if not origin or not dest:return 'unknown'
    if origin==dest:return 'u_turn_or_incomplete'
    d={('W','E'):'straight',('E','W'):'straight',('N','S'):'straight',('S','N'):'straight',
       ('W','N'):'left',('N','E'):'left',('E','S'):'left',('S','W'):'left',
       ('W','S'):'right',('S','E'):'right',('E','N'):'right',('N','W'):'right'}
    return d.get((origin,dest),'unknown')


class Signals:
    def __init__(self,path,config_path=None,offset=0,transition_margin_ms=350,max_age_ms=180000):
        self.offset=float(offset);self.margin=float(transition_margin_ms);self.max_age=float(max_age_ms)
        self.data={};self.time_verified=False;self.rules={};self.file_present=False
        for arm,idx in DEFAULT_LIGHTS.items():
            self.rules[arm] = {'motor':idx,'nonmotor':None,'verified_motor':False,
                               'verified_nonmotor':False,'source':'SinD_tj_mapping_not_video_verified'}
        if config_path:
            j=json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
            self.time_verified=bool(j.get('time_alignment_verified',False))
            for arm,settings in j.get('arms',{}).items():
                if arm not in ARMS:continue
                self.rules[arm].update(settings)
        if not path or not path.is_file():return
        ensure_real_csv(path)
        d=pd.read_csv(path,low_memory=False)
        if 'timestamp(ms)' not in d:raise ValueError('Licht-CSV heeft geen timestamp(ms)')
        time=pd.to_numeric(d['timestamp(ms)'],errors='coerce').to_numpy(float)-self.offset
        if not np.isfinite(time).all():raise ValueError('Ongeldige licht-timestamps')
        for c in d.columns:
            m=re.search(r'(?:vehicle\s+)?traffic\s*light\s*(\d+)\s*$',str(c),re.I)
            if not m:continue
            k=int(m.group(1));raw=pd.to_numeric(d[c],errors='coerce').to_numpy(float)
            good=np.isfinite(raw); order=np.argsort(time[good],kind='stable')
            tt=time[good][order];vv=raw[good][order]
            if not len(tt):continue
            # Bij dubbele transitie-timestamps geldt de laatste rij.
            ix=pd.DataFrame({'time':tt,'value':vv}).drop_duplicates('time',keep='last')
            self.data[k]=(ix.time.to_numpy(float),ix.value.to_numpy(float))
        self.file_present=True
    def applicable(self,arm,vehicle_class):
        if arm not in self.rules:return None,False,'no_origin'
        r=self.rules[arm]
        if vehicle_class=='motorized':
            return r.get('motor'),bool(r.get('verified_motor',False)),'motor_signal'
        if vehicle_class=='non_motorized':
            if r.get('nonmotor') is not None:
                return r['nonmotor'],bool(r.get('verified_nonmotor',False)),'nonmotor_signal'
            # Voertuiglicht enkel CONTEXT, geen zelfstandig bewijs voor fietsregel.
            return r.get('motor'),False,'motor_signal_proxy_for_nonmotor'
        return r.get('motor'),False,'unknown_agent_signal_applicability'
    def at(self,sid,t):
        if sid not in self.data or not np.isfinite(t):return ('unknown',False)
        tt,v=self.data[sid]
        i=int(np.searchsorted(tt,t,side='right')-1)
        if i<0 or t-tt[i]>self.max_age:return ('unknown',False)
        state=LIGHT_CODES.get(int(v[i]),'unknown')
        stable=(state!='unknown' and not np.any(np.abs(tt-t)<=self.margin))
        return state,bool(stable)
    def during(self,sid,start,end,state='red'):
        if sid is None or end<start:return 0.
        if end==start:return float(self.at(sid,start)[0]==state)
        # Transities en intervaleindpunten: exact de duur van red/green/yellow.
        ts=self.data[sid][0] if sid in self.data else np.array([])
        mids=ts[(ts>start)&(ts<end)]
        ends=np.r_[start,mids,end]
        return sum((b-a)/1000 for a,b in zip(ends[:-1],ends[1:])
                   if self.at(sid,(a+b)/2)[0]==state)


def verify_tianjin_phases(signals, strict=True):
    """Replay 0/10/13/14/40/43/44 s, independently observed in recording 8_2_1.

    Checks phase equality across groups over ALL timestamps and expected early
    timeline. Does not establish which individual light/pedestrian head serves
    a given lane. Refuses unsupported interpolation/extrapolation or wrong file.
    """
    groups={'NS':(1,4,5,8),'EW':(2,3,6,7)}
    if not all(k in signals.data for ks in groups.values() for k in ks):
        raise ValueError('Onvolledig Tianjin verkeerslichtbestand: verwachte lichten 1 t/m 8 ontbreken')
    for direction, group in groups.items():
        ref_t,ref_v=signals.data[group[0]]
        for other in group[1:]:
            ts,vs=signals.data[other]
            if len(ts)!=len(ref_t) or not np.array_equal(ts,ref_t) or not np.array_equal(vs,ref_v):
                raise ValueError(f'Verkeerslichtgroepen inconsistent voor {direction}: licht {other}')
    expected=((0,'green','red'),(10000,'yellow','red'),(13000,'red','red'),
              (14000,'red','green'),(40000,'red','yellow'),(43000,'red','red'),
              (44000,'green','red'))
    failures=[]
    for t,ns,ew in expected:
        actual=(signals.at(1,t)[0],signals.at(7,t)[0])
        if actual!=(ns,ew):failures.append({'time_ms':t,'expected':[ns,ew],'actual':list(actual)})
    if failures and strict:
        raise ValueError('Signaaldata komt niet overeen met geverifieerde replayfase: '+json.dumps(failures))
    signals.phase_group_verified=not failures
    signals.time_verified=not failures and abs(signals.offset)<1e-9
    for arm in ARMS:
        signal_id=DEFAULT_LIGHTS[arm]
        signals.rules[arm]['motor']=signal_id
        signals.rules[arm]['verified_motor']=signals.phase_group_verified
        signals.rules[arm]['source']='Tianjin_8_2_1_CSV_phase_groups_replay_verified; individual_head_unverified'
    return {'phase_group_verified':signals.phase_group_verified,
            'timeline_verified':signals.time_verified,'failures':failures,'groups':groups}


def length_from_data(g,md):
    """Use dimensions only if actually recorded. No guessed car/bicycle lengths."""
    for origin in (g.iloc[0].to_dict(),md):
        for key in ('length','length_m','vehicle_length','vehicleLength'):
            v=numeric(origin.get(key))
            if v is not None and .7<=v<=30:return float(v),key
    return None,'not_available'


def width_from_data(g,md):
    for origin in (g.iloc[0].to_dict(),md):
        for key in ('width','width_m','vehicle_width','vehicleWidth'):
            v=numeric(origin.get(key))
            if v is not None and .2<=v<=5:return float(v),key
    return None,'not_available'


def stationary_footprint(g,idx,length_m,width_m):
    """Oriented rectangle ONLY when both real dimensions and approach motion exist.
    Heading is the preceding motion, not a fabricated heading while stopped.
    """
    if length_m is None or width_m is None:return None
    stop=int(idx[0]);begin=max(0,stop-20)
    if stop-begin<2:return None
    xy=g.iloc[begin:stop][['x','y']].to_numpy(float)
    heading=unit(xy[-1]-xy[0])
    if heading is None or np.linalg.norm(xy[-1]-xy[0])<.8:return None
    side=np.array([-heading[1],heading[0]])
    mid=np.median(g.iloc[idx][['x','y']].to_numpy(float),axis=0)
    corners=[]
    for along,across in ((1,1),(1,-1),(-1,-1),(-1,1)):
        corners.append(mid+along*heading*(length_m/2)+across*side*(width_m/2))
    return Polygon(corners)


def classify_events(events):
    """Fuse overlapping wrong-way and cut-corner reports without erasing evidence."""
    priority={'destination_inbound_wrong_side_turn_candidate':2,'wrong_way_road_segment_candidate':1}
    for ev in events:
        typ=ev['event_type'];ev['duplicate_of']=''
        if typ.startswith('green_signal_'):ev['assessment']='control_not_violation'
        elif typ.startswith(('signal_unknown_','track_starts_')):ev['assessment']='not_assessable'
        elif typ=='red_signal_at_reference_crossing':
            if ev['movement']=='right' and ev['vehicle_category']=='motorized':
                ev['assessment']='red_crossing_right_turn_legal_review'
            elif ev['signal_applicability']!='motor_signal' or not ev['phase_stable']:
                ev['assessment']='red_crossing_signal_uncertain'
            else:ev['assessment']='red_crossing_behavior_supported_legal_review'
        elif typ=='wait_before_reference_line_during_red':ev['assessment']='control_not_violation'
        elif typ=='wait_at_reference_line_during_red':ev['assessment']='at_line_uncertain'
        elif typ=='wait_past_reference_line_during_red':ev['assessment']='stopped_beyond_line_candidate'
        elif typ.startswith('yellow_'):ev['assessment']='yellow_crossing_context_review'
        else:ev['assessment']='behavior_candidate_review'
    for a in events:
        if a['event_type']!='destination_inbound_wrong_side_turn_candidate':continue
        for b in events:
            if b['event_type']!='wrong_way_road_segment_candidate':continue
            if a['arm']!=b['arm'] or a['end_ms']<b['start_ms'] or b['end_ms']<a['start_ms']:continue
            b['duplicate_of']='destination_inbound_wrong_side_turn_candidate'
            b['assessment']='same_maneuver_as_wrong_side_turn'
    return events


def vehicle_conclusions(events, category):
    primary=[e for e in events if not e['duplicate_of']]
    names=sorted(set(e['event_type'] for e in primary if e['assessment'] not in ('control_not_violation','not_assessable')))
    if any(e['assessment'] in ('red_crossing_behavior_supported_legal_review','red_crossing_right_turn_legal_review') for e in primary):
        verdict='observed_red_crossing_candidate_not_automatic_legal_violation'
    elif names:verdict='one_or_more_behaviour_candidates_for_review'
    elif any(e['assessment']=='not_assessable' for e in primary):verdict='insufficient_observation_for_some_rules'
    else:verdict='no_candidate_detected_not_proof_of_compliance'
    return verdict,';'.join(names),len(names)


@dataclass
class Gate:
    arm: str
    a: np.ndarray
    b: np.ndarray
    normal: np.ndarray
    validated: bool
    kind: str
    source: str
    @property
    def mid(self):return (self.a+self.b)/2
    def position(self,p):
        p=np.asarray(p,float)
        d=float(np.dot(p-self.mid,self.normal))
        seg=self.b-self.a
        u=float(np.dot(p-self.a,seg)/np.dot(seg,seg))
        return d,u
    def lateral_ok(self,p,margin=0.8):
        _,u=self.position(p)
        # ±0.8 m voorbij eindpunt, niet de volledige oneindige lijn.
        ext=margin/max(1e-8,np.linalg.norm(self.b-self.a))
        return -ext<=u<=1+ext


def make_gates(roads,stopfile,outdir):
    gates={};templ=[]
    for arm in ARMS:
        gaps=[a for a in roads if a.role=='approach_gap' and a.arm==arm and a.flow=='in' and a.direction_status=='geometry_consistent_direction_assumed']
        z=gaps if gaps else [a for a in roads if a.role=='approach_or_departure' and a.arm==arm and a.flow=='in' and a.direction_status=='geometry_consistent_direction_assumed']
        if not z:continue
        direction=unit(INWARD[arm]);perp=np.array([-direction[1],direction[0]])
        ends=np.array([x.end for x in z]); c=ends.mean(axis=0)
        trans=np.dot(ends-c,perp)
        a=c+perp*(float(np.min(trans))-2.0)
        b=c+perp*(float(np.max(trans))+2.0)
        g=Gate(arm,a,b,direction,True,'approach_gap_lanelet_end_assumed_stopline' if gaps else 'road_lanelet_end_assumed_stopline','USER_ASSUMPTION_LANELET_END_IS_STOPLINE')
        gates[arm]=g
        templ.append({'arm':arm,'x1_m':round(a[0],3),'y1_m':round(a[1],3),
                      'x2_m':round(b[0],3),'y2_m':round(b[1],3),
                      'validated':True,'source':'USER_ASSUMPTION_LANELET_END_IS_STOPLINE'})
    pd.DataFrame(templ).to_csv(outdir/'stoplijnen_OP_BASIS_VAN_LANELET_EINDEN.csv',index=False)
    if stopfile:
        ensure_real_csv(stopfile)
        d=pd.read_csv(stopfile)
        required={'arm','x1_m','y1_m','x2_m','y2_m','validated','source'}
        if not required.issubset(d):raise ValueError('Stoplijn-CSV mist: '+str(required-set(d.columns)))
        for row in d.to_dict('records'):
            arm=str(row['arm']).strip().upper()
            if arm not in ARMS:raise ValueError('Onbekende arm stoplijn: '+arm)
            a=np.array([float(row['x1_m']),float(row['y1_m'])]);b=np.array([float(row['x2_m']),float(row['y2_m'])])
            direction=unit(INWARD[arm]);vec=b-a
            if np.linalg.norm(vec)<.5:raise ValueError('Stoplijn te kort voor '+arm)
            if abs(float(np.dot(unit(vec),direction)))>.4:raise ValueError('Stoplijn niet ongeveer haaks op rijrichting: '+arm)
            valid=str(row['validated']).strip().lower() in ('true','yes','1','ja')
            gates[arm]=Gate(arm,a,b,direction,valid,'measured_stopline' if valid else 'user_unverified_line',str(row['source']))
    return gates


def tangent(z,p):
    q=Point(float(p[0]),float(p[1]));L=z.center
    d=L.project(q);lo=L.interpolate(max(0,d-.9));hi=L.interpolate(min(L.length,d+.9))
    return unit((hi.x-lo.x,hi.y-lo.y))


def cosine(z,p,v):
    a=unit(v);b=tangent(z,p)
    return float(np.dot(a,b)) if a is not None and b is not None else float('nan')


def assign_road_points(g,roads_spatial,vel,speed,crosswalks):
    x=g.x.to_numpy(float);y=g.y.to_numpy(float)
    result=[];prev=None
    for i,(a,b) in enumerate(zip(x,y)):
        p=Point(float(a),float(b)); cand=roads_spatial.query(p,.35)
        # Een fysieke lane altijd boven een gele verbinding verkiezen; deze
        # query bevat uitsluitend R...In/Out roadpolygonen.
        scored=[]
        for z,d,cl in cand:
            penalty = 0.0 if z.id==prev else .12
            scored.append((5*d+.18*cl+penalty,z,d,cl))
        scored.sort(key=lambda e:e[0])
        hit=scored[0] if scored else None
        ambiguous=(len(scored)>1 and scored[1][0]-scored[0][0]<.12)
        if hit:
            z=hit[1];prev=z.id
            dc=cosine(z,(a,b),vel[i])
            motion='stationary_or_unknown'
            if np.isfinite(dc) and np.isfinite(speed[i]) and speed[i]>=.70:
                motion='along' if dc>=.60 else ('against' if dc<=-.60 else 'transverse')
            fields={'lanelet_id':z.id,'lanelet_name':z.name,'arm':z.arm,'flow':z.flow,
                    'lane_role':z.lane_role,'direction_cosine':dc,'motion':motion,
                    'assignment_status':'ambiguous_physical_lane' if ambiguous else 'matched_physical_lane',
                    'centerline_distance_m':hit[3]}
        else:
            fields={'lanelet_id':'','lanelet_name':'','arm':'','flow':'','lane_role':'',
                    'direction_cosine':np.nan,'motion':'unassigned',
                    'assignment_status':'outside_physical_road','centerline_distance_m':np.nan}
            prev=None
        crosses=[z for z in crosswalks.query(p,0) if z[0].poly.covers(p)]
        fields['on_crosswalk']=bool(crosses)
        fields['crosswalk_id']=';'.join(z[0].id for z in crosses)
        result.append(fields)
    return result


def episode_groups(mask,t,max_gap_ms=550):
    groups=[];g=[]
    for i,yes in enumerate(mask):
        if not yes:
            if g:groups.append(g);g=[]
            continue
        if g and t[i]-t[g[-1]]>max_gap_ms:
            groups.append(g);g=[]
        g.append(i)
    if g:groups.append(g)
    return groups


def path_length(g,idx):
    if len(idx)<2:return 0.
    xy=g.iloc[idx][['x','y']].to_numpy(float)
    return float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())


def base_event(g,atype,category,od,evtype,idx,**kwargs):
    s=g.iloc[idx[0]];e=g.iloc[idx[-1]]
    v={'track_id':clean_id(s.track_id),'agent_type':atype,'vehicle_category':category,
       'origin':od[0],'destination':od[1],'movement':movement_kind(*od),
       'event_type':evtype,'evidence_level':'candidate','review_status':'manual_review_required',
       'reason':'','zone':'','arm':'','start_ms':float(s.timestamp_ms),'end_ms':float(e.timestamp_ms),
       'duration_s':round(float(e.timestamp_ms-s.timestamp_ms)/1000,3),
       'samples':len(idx),'travel_distance_m':round(path_length(g,idx),3),
       'lanelet_ids':'','x_m':round(float(g.iloc[idx].x.mean()),3),
       'y_m':round(float(g.iloc[idx].y.mean()),3),
       'signal_id':'','signal_state':'','signal_applicability':'',
       'signal_mapping_verified':False,'time_alignment_verified':False,'phase_stable':'',
       'line_type':'','line_validated':False,'reference_point':'tracked_position_unknown_reference',
       'detail':'','duplicate_of':'','assessment':'review_required'}
    v.update(kwargs)
    return v


def lane_events(g,assigned,vel,speed,atype,category,od,min_distance=2.0,min_samples=3):
    t=g.timestamp_ms.to_numpy(float);events=[]
    for evtype in ['wrong_way_road_segment_candidate','nonmotor_in_assumed_motor_lane',
                   'motor_in_assumed_nonmotor_lane','destination_inbound_wrong_side_turn_candidate']:
        mask=[]
        for a in assigned:
            if a['assignment_status']!='matched_physical_lane':mask.append(False);continue
            dest_inbound=bool(od[1] and od[0] and od[0]!=od[1] and a['arm']==od[1] and a['flow']=='in')
            matched=(evtype=='wrong_way_road_segment_candidate' and a['motion']=='against' or
                     evtype=='nonmotor_in_assumed_motor_lane' and category=='non_motorized' and a['motion']=='along' and a['lane_role']=='motor_vehicle' or
                     evtype=='motor_in_assumed_nonmotor_lane' and category=='motorized' and a['motion']=='along' and a['lane_role']=='non_motorized' or
                     evtype=='destination_inbound_wrong_side_turn_candidate' and dest_inbound and a['motion']=='against')
            mask.append(bool(matched))
        for idx in episode_groups(mask,t):
            if len(idx)<min_samples or path_length(g,idx)<min_distance:continue
            # Dwing dezelfde fysieke arm + bewegingsrichting af: tegen Lanelet
            # rijden ≠ een kruisende Lanelet kort raken.
            lane_ids=set(assigned[i]['lanelet_id'] for i in idx)
            arms=set(assigned[i]['arm'] for i in idx)
            if len(arms)>1:continue
            arm=assigned[idx[0]]['arm'];firstflow=assigned[idx[0]]['flow']
            zon='entry' if arm==od[0] and firstflow=='in' else ('exit' if arm==od[1] and firstflow=='out' else 'turn_or_other')
            reason={'wrong_way_road_segment_candidate':'trajectory_moves_along_physical_road_against_nominal_direction',
                    'nonmotor_in_assumed_motor_lane':'assumed_lane_role_not_legally_verified',
                    'motor_in_assumed_nonmotor_lane':'assumed_lane_role_not_legally_verified',
                    'destination_inbound_wrong_side_turn_candidate':'trajectory_follows_destination_inbound_lane_against_its_direction_not_just_crossing'}[evtype]
            events.append(base_event(g,atype,category,od,evtype,idx,arm=arm,zone=zon,
                   lanelet_ids=';'.join(sorted(lane_ids)),reason=reason))
    return events



def turning_and_route_events(g,assigned,vel,speed,lanes,atype,category,od):
    """Aanvullende fysieke observaties, expliciet GEEN wettelijke overtredingen.

    1) Controleer rijstrookpijlen bij het verlaten van de herkomstarm.
    2) Controleer werkelijk gereden middensegment t.o.v. gemodelleerde
       OD-verbindingspolygonen; modelafwijking is geen bewijs van illegale route.
    """
    origin,dest=od
    if not origin or not dest or origin==dest:return []
    move=movement_kind(origin,dest)
    t=g.timestamp_ms.to_numpy(float)
    ev=[]
    if category=='motorized' and move in ('left','right','straight'):
        roads_by_id={z.id:z for z in lanes if z.role=='approach_or_departure'}
        mask=[bool(a['arm']==origin and a['flow']=='in' and a['motion']=='along' and
                   a['assignment_status']=='matched_physical_lane' and a['lanelet_id'] in roads_by_id)
              for a in assigned]
        groups=episode_groups(mask,t)
        groups=[ids for ids in groups if len(ids)>=3 and path_length(g,ids)>=2.5]
        # Alleen de duidelijkste stabiele instrook, zonder interpretatie van
        # toevallige Lanelets in het centrum als rechtsgeldige pijlmarkering.
        if groups:
            ids=max(groups,key=lambda ids:path_length(g,ids))
            lane_id=Counter(assigned[i]['lanelet_id'] for i in ids).most_common(1)[0][0]
            z=roads_by_id[lane_id]
            allowed={x.strip().lower() for x in str(z.marked_movements).split(';')}
            if allowed and 'unspecified' not in allowed and 'unknown' not in allowed and move not in allowed:
                ev.append(base_event(g,atype,category,od,'entry_lane_marking_incompatible_with_turn_candidate',
                   ids,arm=origin,zone='entry',lanelet_ids=lane_id,
                   reason='model_assumes_arrow_marking_and_lane_assignment_are_correct',
                   detail=f'observed_movement={move};mapped_arrows={z.marked_movements}'))
    opts=[z for z in lanes if z.role=='lane_specific_connector' and z.from_arm==origin and z.to_arm==dest]
    if category=='motorized':
        typed=[z for z in opts if z.lane_role=='motor_vehicle']
        if typed:opts=typed
    elif category=='non_motorized':
        typed=[z for z in opts if z.lane_role=='non_motorized']
        if typed:opts=typed
    if not opts:return ev
    # Een bounding box over gemodelleerde OD-verbindingen (geen gebied buiten
    # het kruispunt). Geen koppeling op basis van willekeurige andere Lanelets.
    lo=np.min([z.poly.bounds[:2] for z in opts],axis=0)
    hi=np.max([z.poly.bounds[2:] for z in opts],axis=0)
    xy=g[['x','y']].to_numpy(float)
    mask=[]
    for i,p in enumerate(xy):
        if not np.isfinite(speed[i]) or speed[i]<.7:mask.append(False);continue
        if np.any(p<lo+1.0) or np.any(p>hi-1.0):mask.append(False);continue
        if assigned[i]['arm']==origin and assigned[i]['flow']=='in':mask.append(False);continue
        if assigned[i]['arm']==dest and assigned[i]['flow']=='out':mask.append(False);continue
        distance=min(z.poly.distance(Point(float(p[0]),float(p[1]))) for z in opts)
        mask.append(distance>2.5)
    for ids in episode_groups(mask,t):
        if len(ids)<4 or path_length(g,ids)<3.0:continue
        ev.append(base_event(g,atype,category,od,'outside_modelled_OD_connector_corridor_candidate',
           ids,zone='intersection',reason='off_model_geometry_not_necessarily_illegal_two_stage_turns_or_missing_routes',
           detail='geometric_buffer_2.5m;OD_specific_connector_hypothesis_not_a_legal_standard'))
    return ev


def gate_crossings(g,gate,signals,sid,verified,mapping_type,atype,category,od,transition_ms,length_m=None):
    if not gate or not od[0]:return []
    xy=g[['x','y']].to_numpy(float);t=g.timestamp_ms.to_numpy(float)
    d=np.array([gate.position(p)[0] for p in xy]);events=[]
    for i in range(1,len(g)):
        dt=t[i]-t[i-1]
        if not (0<dt<=700):continue
        delta=xy[i]-xy[i-1];heading=unit(delta)
        forward=float(np.dot(heading,gate.normal)) if heading is not None else 0.
        if forward<.65:continue
        # In SinD x/y is normally an object reference point; use length only if recorded.
        # Constant segment heading prevents an artificial sign change from a heading jump.
        offset=.5*length_m*forward if length_m is not None else 0.
        d0,d1=d[i-1]+offset,d[i]+offset
        if not (d0<0<=d1 and d1-d0>=.25):continue
        f=float(-d0/(d1-d0))
        cross=xy[i-1]+f*(xy[i]-xy[i-1]); tc=float(t[i-1]+f*dt)
        if not gate.lateral_ok(cross):continue
        phase,stable=signals.at(sid,tc)
        movement=movement_kind(*od)
        if phase=='red':
            typ='red_signal_at_reference_crossing'
            if movement=='right' and category=='motorized':review='right_on_red_conditional_manual_yield_and_arrow_review'
            else:review='manual_review_required'
        elif phase=='yellow':typ='yellow_signal_at_reference_crossing';review='yellow_rule_review'
        elif phase=='green':typ='green_signal_at_reference_crossing';review='observational_control'
        else:typ='signal_unknown_at_reference_crossing';review='not_assessable'
        evidence='candidate'
        if (gate.validated and verified and signals.time_verified and stable
            and category!='uncertain' and mapping_type!='motor_signal_proxy_for_nonmotor'):
            evidence='geometrically_and_temporally_supported_candidate'
        if phase=='unknown' or not stable:evidence='uncertain'
        ev=base_event(g,atype,category,od,typ,[i-1,i],arm=gate.arm,zone='entry',
                      reason='front_point_from_measured_length_and_motion' if length_m is not None else 'tracked_position_no_length_available',
                      signal_id=sid or '',signal_state=phase,
                      signal_applicability=mapping_type,
                      signal_mapping_verified=verified,time_alignment_verified=signals.time_verified,
                      phase_stable=stable,line_type=gate.kind,line_validated=gate.validated,
                      x_m=round(float(cross[0]),3),y_m=round(float(cross[1]),3),
                      start_ms=round(tc,2),end_ms=round(tc,2),duration_s=0,
                      travel_distance_m=round(float(np.linalg.norm(xy[i]-xy[i-1])),3),
                      evidence_level=evidence,review_status=review,
                      reference_point='estimated_front_from_recorded_length' if length_m is not None else 'tracked_position_unknown_reference',
                      detail=f'phase_transition_margin_ms={transition_ms};vehicle_length_m={length_m};right_on_red_conditional_if_applicable')
        events.append(ev)
    return events


def stopping_episodes(g,vel,speed,threshold=.40,min_seconds=1.2,max_drift=1.5):
    t=g.timestamp_ms.to_numpy(float)
    xy=g[['x','y']].to_numpy(float)
    # Voorkom een los stilstaand punt door minimale duur én kleine spreiding.
    mask=np.isfinite(speed)&(speed<=threshold)
    result=[]
    for idx in episode_groups(mask,t,max_gap_ms=500):
        if len(idx)<3 or (t[idx[-1]]-t[idx[0]])/1000 < min_seconds:continue
        a=xy[idx]
        if np.max(np.linalg.norm(a-np.median(a,axis=0),axis=1))>max_drift:continue
        result.append(idx)
    return result


def stop_events(g,assigned,stops,gate,signals,sid,verified,mapping_type,atype,category,od,length_m=None,width_m=None,crosswalks=None):
    events=[]
    for idx in stops:
        p=np.median(g.iloc[idx][['x','y']].to_numpy(float),axis=0)
        start=float(g.timestamp_ms.iloc[idx[0]]);end=float(g.timestamp_ms.iloc[idx[-1]])
        red_dur=signals.during(sid,start,end,'red')
        arm=od[0] or ''
        # Stilstaan op zebrapad is een eigen geometrische observatie,
        # ook zonder valide licht of stopstreep.
        cs=[a['crosswalk_id'] for a in (assigned[i] for i in idx) if a['on_crosswalk']]
        footprint=stationary_footprint(g,idx,length_m,width_m)
        if footprint is not None and crosswalks is not None:
            for z in crosswalks:
                if footprint.intersects(z.poly) and z.id not in cs:cs.append(z.id)
        on_crosswalk=len(cs)/len(idx)>=.5 or (footprint is not None and bool(cs))
        if on_crosswalk:
            evtype='stationary_on_crosswalk_during_motor_signal_red' if red_dur>=.5 else 'stationary_on_crosswalk'
            events.append(base_event(g,atype,category,od,evtype,idx,zone='crosswalk',
               arm=arm,signal_id=sid or '',signal_state='red' if red_dur>=.5 else 'unknown_or_not_red',
               signal_applicability=mapping_type,signal_mapping_verified=verified,
               time_alignment_verified=signals.time_verified,lanelet_ids=';'.join(sorted(set(';'.join(cs).split(';')))),
               reason='measured_dimension_vehicle_footprint_intersects_crosswalk' if footprint is not None else 'tracked_reference_point_on_crosswalk_vehicle_extent_unknown',
               detail=f'red_overlap_s={red_dur:.3f}'))
        if not gate or not gate.lateral_ok(p) or red_dur<.5:continue
        d,_=gate.position(p)
        if abs(d)>12:continue
        # Stoppositie: voorbumper beter dan middelpunt als lengte werkelijk beschikbaar.
        # Gebruik netto aanrijrichting van maximaal 20 eerdere punten voor heading.
        reference='tracked_position_unknown_reference'
        if length_m is not None:
            ix0=max(0,int(idx[0])-20);ix1=max(ix0,int(idx[0])-1)
            before=g.iloc[ix0:ix1+1][['x','y']].to_numpy(float)
            if len(before)>=2:
                heading=unit(before[-1]-before[0])
                cos=float(np.dot(heading,gate.normal)) if heading is not None else 0.
                if cos>.65:
                    d+=.5*length_m*cos
                    reference='estimated_front_from_recorded_length'

        # +/- 0.7 m is de onzekerheidsbuffer voor de referentiepuntpositie.
        if d>.7:kind='wait_past_reference_line_during_red'
        elif d<-.7:kind='wait_before_reference_line_during_red'
        else:kind='wait_at_reference_line_during_red'
        events.append(base_event(g,atype,category,od,kind,idx,arm=arm,zone='entry',
             reason='stopline_lanelet_end_user_assumption_and_front_estimate_if_available',
             signal_id=sid or '',signal_state='red',signal_applicability=mapping_type,
             signal_mapping_verified=verified,time_alignment_verified=signals.time_verified,
             line_type=gate.kind,line_validated=gate.validated,
             evidence_level='geometric_candidate' if gate.validated else 'proxy_only',
             reference_point=reference,
             detail=f'signed_distance_m={d:.3f};red_overlap_s={red_dur:.3f};front_reference={reference}'))
    return events


def censor_event(g,gate,atype,category,od,events,length_m=None):
    if not gate or not od[0] or any('reference_crossing' in e['event_type'] for e in events):return []
    p=g.iloc[0]
    dist,lateral=gate.position((p.x,p.y))
    if length_m is not None and len(g)>1:
        vec=unit(g.iloc[1][['x','y']].to_numpy(float)-g.iloc[0][['x','y']].to_numpy(float))
        projected=float(np.dot(vec,gate.normal)) if vec is not None else 0.
        if projected>.65:dist+=.5*length_m*projected
    if .7<dist<8 and gate.lateral_ok((p.x,p.y)):
        return [base_event(g,atype,category,od,'track_starts_past_reference_line_observation_incomplete',[0],
                           arm=od[0],zone='entry',line_type=gate.kind,line_validated=gate.validated,
                           evidence_level='not_assessable',review_status='cannot_infer_previous_red_crossing',
                           reason='left_censored_track_starts_downstream_of_reference')]
    return []


def connector_evidence(g,lanes,od,vel,speed,atype):
    opts=[z for z in lanes if z.role=='lane_specific_connector' and z.from_arm==od[0] and z.to_arm==od[1]]
    t=g.timestamp_ms.to_numpy(float);xy=g[['x','y']].to_numpy(float)
    results=[]
    for z in opts:
        near=[];aligned=[];reverse=[];transverse=[]
        if z.direction_status=='reversed_geometry_do_not_use_for_direction':
            results.append({'track_id':clean_id(g.track_id.iloc[0]),'origin':od[0],'destination':od[1],
              'connector_id':z.id,'connector_name':z.name,'matched_points':0,'aligned_points':0,
              'reverse_points':0,'transverse_points':0,'along_distance_m':0.,
              'status':'reversed_geometry_direction_unverified'});continue
        for i,p in enumerate(xy):
            if z.poly.distance(Point(*p))>.35:continue
            near.append(i)
            c=cosine(z,p,vel[i])
            if not np.isfinite(c) or not np.isfinite(speed[i]) or speed[i]<.7:continue
            if c>=.6:aligned.append(i)
            elif c<=-.6:reverse.append(i)
            else:transverse.append(i)
        aligned_path=0.
        for a,b in zip(aligned[:-1],aligned[1:]):
            if b==a+1 and t[b]-t[a]<=550:aligned_path+=np.linalg.norm(xy[b]-xy[a])
        status=('plausibly_followed_OD_connector' if len(aligned)>=3 and aligned_path>=2.5 else
                'overlap_not_evidence_of_following' if near else 'not_observed')
        results.append({'track_id':clean_id(g.track_id.iloc[0]),'origin':od[0],'destination':od[1],
          'connector_id':z.id,'connector_name':z.name,'matched_points':len(near),'aligned_points':len(aligned),
          'reverse_points':len(reverse),'transverse_points':len(transverse),
          'along_distance_m':round(float(aligned_path),3),'status':status})
    return results


def make_plot(g,roads,crosswalks,gates,events,od,atype,out,image_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Patch
    fig,ax=plt.subplots(figsize=(11,8))
    img=None
    if image_path and Path(image_path).is_file():
        img=plt.imread(image_path)
    def tf(v):
        xy=np.asarray(v,dtype=float)
        if img is None:return xy
        k=img.shape[1]/2576.0
        return np.column_stack(((725.7+35.05*xy[:,0])*k,
                                (1322.0-35.05*xy[:,1])*k))
    if img is not None:ax.imshow(img)
    for z in roads:
        ax.add_patch(Patch(tf(z.poly.exterior.coords),fill=False,edgecolor='#8b919c',lw=.85))
    for z in crosswalks:
        ax.add_patch(Patch(tf(z.poly.exterior.coords),fill=True,facecolor='#d8c9ee',edgecolor='#9b75b1',alpha=.27,lw=.8))
    for arm,gate in gates.items():
        pts=tf(np.vstack((gate.a,gate.b,gate.mid)))
        ax.plot(pts[:2,0],pts[:2,1],'--',lw=1.6,
                color='#2478ab' if gate.validated else '#d97706')
        ax.text(pts[2,0],pts[2,1],arm,fontsize=8)
    xy=tf(g[['x','y']].to_numpy(float))
    ax.plot(xy[:,0],xy[:,1],c='#2563eb',lw=1.9)
    ax.scatter(xy[0,0],xy[0,1],c='green',s=35,label='start')
    ax.scatter(xy[-1,0],xy[-1,1],c='red',s=35,label='eind')
    for ev in events:
        if 'reference_crossing' in ev['event_type'] or 'wrong_side_turn' in ev['event_type'] or 'crosswalk' in ev['event_type']:
            q=tf([[ev['x_m'],ev['y_m']]])[0]
            ax.scatter([q[0]],[q[1]],c='#eab308',s=42,zorder=7)
    ax.set_title(f'Track {clean_id(g.track_id.iloc[0])} ({atype}) {od[0]}→{od[1]} | KANDIDATEN, GEEN VASTGESTELDE OVERTREDINGEN')
    ax.set_aspect('equal')
    if img is not None:
        ax.set_xlim(0,img.shape[1]);ax.set_ylim(img.shape[0],0)
        ax.set_xlabel('luchtfoto x (px)');ax.set_ylabel('luchtfoto y (px)')
    else:ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    ax.legend();fig.tight_layout()
    fig.savefig(out/f'track_{clean_id(g.track_id.iloc[0])}.png',dpi=140);plt.close(fig)


def export_config(out,lanes,dropped,gates,signals,counts,options):
    config={'prototype':'v3','notes':[
      'Geen bewezen juridische overtredingen; zie handmatige validatie en lokale regelgeving.',
      'Motorvoertuigsignaal geldt niet automatisch voor fietsers.',
      'Stoplijn = einde korte approach_gap-Lanelets (indien aanwezig), door gebruiker als stoplijn aangenomen; niet onafhankelijk ingemeten.',
      'Voertuigvoorkant alleen benaderd indien gemeten voertuiglente beschikbaar; voertuigoriëntatie uit beweging.',
      'Lanelet-functies zijn werkhypothesen; geel is niet automatisch juridisch fietsnetwerk.',
      'Rechtsaf bij rood is mogelijk onder voorwaarden; niet automatisch overtreding.',
      'Een voertuig zonder crossing_event kan links-gecensureerd zijn of buiten gate passeren.'
    ],'record_counts':counts,'lanelets':len(lanes),'skipped':dropped,
    'signal_file_present':signals.file_present,'signal_rules':signals.rules,
    'phase_group_verified':getattr(signals,'phase_group_verified',False),
    'time_alignment_verified':signals.time_verified,'signal_offset_ms':signals.offset,
    'gates':{a:{'start':g.a.tolist(),'end':g.b.tolist(),'type':g.kind,'validated':g.validated,'source':g.source}
             for a,g in gates.items()},'run_options':{k:str(v) for k,v in vars(options).items()}}
    (out/'controle_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tracks',type=Path,default=Path('Data/Tianjin/8_2_1/Veh_smoothed_tracks.csv'))
    p.add_argument('--meta',type=Path,default=Path('Data/Tianjin/8_2_1/Veh_tracks_meta.csv'))
    p.add_argument('--signals',type=Path,default=Path('Data/Tianjin/8_2_1/TrafficLight_8_2_1.csv'))
    p.add_argument('--osm',type=Path,default=Path(__file__).resolve().parent/'tianjin_lanelets_uitgebreid.osm')
    p.add_argument('--output-dir',type=Path,default=Path('analysis_outputs/lanelet_audit/prototype_alle_voertuigen_v3'))
    p.add_argument('--stoplines',type=Path,default=None)
    p.add_argument('--no-replay-signal-check',action='store_true',help='UITSLUITEND ontwikkeltest; zonder fasecontrole geen betrouwbare lichtconclusies')
    p.add_argument('--image',type=Path,default=Path('Data/Tianjin/tianjin.jpg'),
                   help='Optionele SinD-luchtfoto voor plots; pixeltransformatie uit eerdere kaart-audit')
    p.add_argument('--signal-config',type=Path,default=None)
    p.add_argument('--signal-offset-ms',type=float,default=0.)
    p.add_argument('--plots',action='store_true')
    p.add_argument('--plot-ids',type=str,default='1,2,4,9,10,11,12,17')
    p.add_argument('--plot-limit',type=int,default=25)
    p.add_argument('--min-episode-m',type=float,default=2.0)
    p.add_argument('--stop-speed-mps',type=float,default=.4)
    p.add_argument('--stop-min-sec',type=float,default=1.2)
    args=p.parse_args(argv)
    ensure_real_csv(args.tracks)
    if not args.osm.is_file():p.error('Uitgebreide Lanelet-kaart ontbreekt: '+str(args.osm))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    lanes,dropped=read_map(args.osm)
    roads=[z for z in lanes if z.role in ('approach_or_departure','approach_gap')]
    crosswalks=[z for z in lanes if z.role=='crosswalk']
    if len(roads)<16:raise ValueError('Lanelet2-kaart: te weinig fysieke wegvakken')
    roadtree=Spatial(roads);cw_tree=Spatial(crosswalks)
    gates=make_gates(roads,args.stoplines,args.output_dir)
    ensure_real_csv(args.signals)
    signals=Signals(args.signals,args.signal_config,args.signal_offset_ms)
    signal_checks=verify_tianjin_phases(signals,strict=not args.no_replay_signal_check)
    tracks,meta=read_tracks(args.tracks,args.meta)
    summaries=[];events=[];samples=[];conn=[]
    plotids={x.strip() for x in args.plot_ids.split(',') if x.strip()};plotted=0
    n_total=len(tracks)
    for tid,g in tracks.groupby('track_id',sort=False):
        g=g.reset_index(drop=True)
        atype=str(g.agent_type.iloc[0]).strip() if g.agent_type.astype(str).str.strip().any() else str(meta.get(tid,{}).get('class','unknown'))
        if atype.lower() in ('','nan','none'):atype=str(meta.get(tid,{}).get('class','unknown'))
        category=role_class(atype)
        vel,speeds=kinematics(g)
        origin,osupport,om=infer_arm(g,roadtree,'origin');dest,dsupport,dm=infer_arm(g,roadtree,'destination')
        od=(origin,dest);move=movement_kind(*od)
        od_status=('classified' if origin and dest and origin!=dest else
                   'same_arm_or_incomplete' if origin and dest else 'incomplete_or_ambiguous')
        assigned=assign_road_points(g,roadtree,vel,speeds,cw_tree)
        ev=lane_events(g,assigned,vel,speeds,atype,category,od,args.min_episode_m)
        ev.extend(turning_and_route_events(g,assigned,vel,speeds,lanes,atype,category,od))
        sid,map_verified,applicability=signals.applicable(origin,category)
        gate=gates.get(origin)
        vehicle_length,length_source=length_from_data(g,meta.get(tid,{}))
        vehicle_width,width_source=width_from_data(g,meta.get(tid,{}))
        crossings=gate_crossings(g,gate,signals,sid,map_verified,applicability,
                                  atype,category,od,signals.margin,length_m=vehicle_length)
        ev.extend(crossings)
        stops=stopping_episodes(g,vel,speeds,args.stop_speed_mps,args.stop_min_sec)
        ev.extend(stop_events(g,assigned,stops,gate,signals,sid,map_verified,applicability,
                              atype,category,od,length_m=vehicle_length,width_m=vehicle_width,crosswalks=crosswalks))
        ev.extend(censor_event(g,gate,atype,category,od,crossings,length_m=vehicle_length))
        classify_events(ev)
        verdict,types,n_types=vehicle_conclusions(ev,category)
        trace=connector_evidence(g,lanes,od,vel,speeds,atype) if od_status=='classified' else []
        conn.extend(trace)
        # Bewijs van routeafwijking alleen als gebrek aan overeenkomst, niet 'verkeersovertreding'.
        # Onderbemonstering en alternatieve tweestapsroutes mogen niet als fout worden gemarkeerd.
        matching=[r for r in trace if r['status']=='plausibly_followed_OD_connector']
        route_status=('plausibly_followed_mapped_OD_connector' if matching else
                      'no_sufficient_connector_evidence' if trace else 'not_assessable')
        for i,r in enumerate(g.itertuples(index=False)):
            a=assigned[i]
            samples.append({'track_id':tid,'agent_type':atype,'timestamp_ms':float(r.timestamp_ms),
                 'x':float(r.x),'y':float(r.y),
                 'speed_mps':round(float(speeds[i]),3) if np.isfinite(speeds[i]) else '',
                 **{k:(round(float(v),3) if isinstance(v,(float,np.floating)) and np.isfinite(v) else v)
                    for k,v in a.items() if k in POINT_COLUMNS}})
        md=meta.get(tid,{})
        summaries.append({'track_id':tid,'agent_type':atype,'vehicle_category':category,
          'n_points':len(g),'start_ms':float(g.timestamp_ms.iloc[0]),'end_ms':float(g.timestamp_ms.iloc[-1]),
          'origin':origin,'destination':dest,'movement':move,'od_status':od_status,
          'origin_support':osupport,'destination_support':dsupport,
          'origin_method':om,'destination_method':dm,
          'signal_id':sid or '','signal_applicability':applicability,
          'signal_mapping_verified':map_verified,'time_alignment_verified':signals.time_verified,
          'gate_type':gate.kind if gate else 'missing','gate_validated':gate.validated if gate else False,
          'reference_crossings':len(crossings),
          'vehicle_length_m':vehicle_length if vehicle_length is not None else '',
          'vehicle_length_source':length_source,
          'vehicle_width_m':vehicle_width if vehicle_width is not None else '',
          'vehicle_width_source':width_source,'classification':verdict,
          'candidate_behavior_types':types,'n_unique_behavior_types':n_types,
          'n_independent_candidates':sum(bool(not e['duplicate_of'] and e['assessment'] not in ('control_not_violation','not_assessable')) for e in ev),
          'red_signal_reference_crossings':sum(e['signal_state']=='red' for e in crossings),
          'red_crossings_stable_phase':sum(e['signal_state']=='red' and e['phase_stable'] for e in crossings),
          'stopping_episodes':len(stops),
          'stops_past_reference_during_red':sum(e['event_type']=='wait_past_reference_line_during_red' for e in ev),
          'stopped_on_crosswalk_episodes':sum('stationary_on_crosswalk' in e['event_type'] for e in ev),
          'wrong_way_episodes':sum(e['event_type']=='wrong_way_road_segment_candidate' for e in ev),
          'sharp_turn_wrong_side_episodes':sum(e['event_type']=='destination_inbound_wrong_side_turn_candidate' for e in ev),
          'turn_from_incompatible_arrow_episodes':sum(e['event_type']=='entry_lane_marking_incompatible_with_turn_candidate' for e in ev),
          'outside_modelled_OD_connector_episodes':sum(e['event_type']=='outside_modelled_OD_connector_corridor_candidate' for e in ev),
          'assumed_wrong_lane_episodes':sum(e['event_type'] in ('nonmotor_in_assumed_motor_lane','motor_in_assumed_nonmotor_lane') for e in ev),
          'route_status':route_status,'relevant_connector_ids':';'.join(z['connector_id'] for z in trace),
          'followed_connector_candidates':';'.join(z['connector_id'] for z in matching),
          'assigned_physical_points':sum(bool(x['lanelet_id']) for x in assigned),
          'ambiguous_physical_points':sum(x['assignment_status']=='ambiguous_physical_lane' for x in assigned),
          'meta_class':md.get('class',''),'meta_Signal_Violation_Behavior':md.get('Signal_Violation_Behavior',''),
          'legal_result':'NOT_DETERMINED_REQUIRES_RULES_AND_VALIDATION'})
        events.extend(ev)
        if args.plots and (tid in plotids or (not plotids and plotted<args.plot_limit)):
            outdir=args.output_dir/'plots';outdir.mkdir(exist_ok=True)
            make_plot(g,roads,crosswalks,gates,ev,od,atype,outdir,args.image)
            plotted+=1
    pd.DataFrame(summaries).to_csv(args.output_dir/'per_voertuig.csv',index=False,encoding='utf-8-sig')
    # Metadata-annotaties dienen UITSLUITEND voor externe controle.
    # Geen validiteitsclaim over overeenstemming: ander definitiekader/stoplijn.
    validation=[]
    for r in summaries:
        annotation=r.get('meta_Signal_Violation_Behavior','')
        if annotation is None or not str(annotation).strip():continue
        validation.append({'track_id':r['track_id'],'agent_type':r['agent_type'],
           'meta_signal_annotation':annotation,
           'prototype_red_at_reference':r['red_signal_reference_crossings']>0,
           'prototype_has_validated_stopline':r['gate_validated'],
           'prototype_signal_applicability':r['signal_applicability'],
           'comparison_status':'diagnostic_not_ground_truth_validation'})
    pd.DataFrame(validation,columns=['track_id','agent_type','meta_signal_annotation',
       'prototype_red_at_reference','prototype_has_validated_stopline',
       'prototype_signal_applicability','comparison_status']).to_csv(
          args.output_dir/'vergelijking_metadata_signalen.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(events,columns=EVENT_COLUMNS).to_csv(args.output_dir/'kandidaat_gebeurtenissen.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(samples,columns=POINT_COLUMNS).to_csv(args.output_dir/'rijstrook_punten.csv',index=False,encoding='utf-8-sig')
    # Geselecteerde, reproduceerbare videoreview-cases (niet als onafhankelijke labels).
    critical={'red_signal_at_reference_crossing','wait_past_reference_line_during_red',
              'stationary_on_crosswalk_during_motor_signal_red',
              'destination_inbound_wrong_side_turn_candidate','wrong_way_road_segment_candidate',
              'entry_lane_marking_incompatible_with_turn_candidate'}
    pd.DataFrame([{'track_id':e['track_id'],'agent_type':e['agent_type'],
       'origin':e['origin'],'destination':e['destination'],'start_ms':e['start_ms'],
       'event_type':e['event_type'],'evidence_level':e['evidence_level'],
       'reason':e['reason']} for e in events if e['event_type'] in critical],
       columns=['track_id','agent_type','origin','destination','start_ms','event_type',
                'evidence_level','reason']).to_csv(args.output_dir/'prioritaire_videocontroles.csv',
                                                 index=False,encoding='utf-8-sig')
    pd.DataFrame(conn,columns=['track_id','origin','destination','connector_id','connector_name','matched_points',
       'aligned_points','reverse_points','transverse_points','along_distance_m','status']).to_csv(
       args.output_dir/'route_evidentie.csv',index=False,encoding='utf-8-sig')
    d=pd.DataFrame(summaries)
    d.groupby(['agent_type','origin','destination','od_status'],dropna=False).size().reset_index(name='n_tracks').to_csv(
        args.output_dir/'od_matrix_alle_voertuigen.csv',index=False,encoding='utf-8-sig')
    counts={'tracks':len(summaries),'points':n_total,'by_agent':dict(Counter(r['agent_type'] for r in summaries)),
            'by_event':dict(Counter(r['event_type'] for r in events)),
            'by_od_status':dict(Counter(r['od_status'] for r in summaries))}
    export_config(args.output_dir,lanes,dropped,gates,signals,counts,args)
    print('Tracks:',counts['tracks'],'punten:',counts['points'],'types:',counts['by_agent'])
    print('OD:',counts['by_od_status'])
    print('Gebeurtenissen:',counts['by_event'])
    print('Gates:',{k:v.kind for k,v in gates.items()})
    print('Signaalsynchronisatie geverifieerd?',signals.time_verified)
    print('Uitvoer:',args.output_dir.resolve())
    print('LET OP: gedragscategorieën/kandidaten, geen juridisch vastgestelde overtredingen; controleer fietslichten en afslagregels.')
    return counts

if __name__=='__main__':main()
