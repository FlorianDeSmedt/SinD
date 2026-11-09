"""Regression tests. Run: python -m unittest -v test_core_v4.py"""
import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from shapely.geometry import Point
from analyse_tianjin_core_v4 import (lanelet_stop_gates, endpoint_arm, crossings,
  wrong_way_events, wait_events, DEFAULTS, INWARD, ARMS, Signals,
  read_map, Spatial, verify_tianjin_phases, kinematics, stopping_episodes,
  movement_kind)

HERE=Path(__file__).resolve().parent
LANES,_=read_map(HERE/'tianjin_lanelets_uitgebreid.osm')
ROADS=[z for z in LANES if z.role in ('approach_gap','approach_or_departure')]
PHY,VIR,AUDIT=lanelet_stop_gates(ROADS)
ROADTREE=Spatial(ROADS)
SIGNALS=Signals(HERE/'TrafficLight_8_2_1.csv')
verify_tianjin_phases(SIGNALS)


def route_xy(x,y,time0=5000,step=100,atype='car'):
    n=len(x)
    return pd.DataFrame(dict(track_id=['TEST']*n,agent_type=[atype]*n,
        timestamp_ms=time0+np.arange(n)*step,x=x,y=y))


def crossing_ew(time0=5000,step=.1,move='straight',size=None):
    # E approach y=21: E stopline x~31.25; x decreasing is inward.
    x=np.linspace(32.6,29.7,30);y=np.full(30,21.0)
    g=route_xy(x,y,time0,int(step*1000))
    vel,speed=kinematics(g)
    dest={'straight':'W','right':'N','left':'S','unknown':''}[move]
    ev=crossings(g,PHY['E'],VIR['E'],SIGNALS,7,'E',dest,vel,size,DEFAULTS)
    return ev,g,vel,speed

class CoreTests(unittest.TestCase):
    def test_signal_data_groups(self):
        self.assertTrue(SIGNALS.phase_group_verified)
        self.assertEqual(SIGNALS.at(7,5000)[0],'red')
        self.assertEqual(SIGNALS.at(1,5000)[0],'green')
        self.assertEqual(SIGNALS.at(7,15000)[0],'green')

    def test_all_arms_both_halves_virtual(self):
        for arm in ARMS:
            self.assertTrue(AUDIT[AUDIT.arm==arm].virtual_covers_center.all(),arm)
            self.assertGreater(np.linalg.norm(VIR[arm].b-VIR[arm].a),np.linalg.norm(PHY[arm].b-PHY[arm].a))

    def test_red_straight(self):
        events,*_=crossing_ew()
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['event_type'],'red_light_running_candidate')

    def test_right_turn_red_exempt(self):
        events,*_=crossing_ew(move='right')
        self.assertEqual(events[0]['event_type'],'right_turn_on_red')

    def test_left_turn_red(self):
        events,*_=crossing_ew(move='left')
        self.assertEqual(events[0]['event_type'],'red_light_running_candidate')

    def test_unknown_turn_red_not_violation(self):
        events,*_=crossing_ew(move='unknown')
        self.assertEqual(events[0]['event_type'],'red_crossing_movement_unknown')

    def test_slow_crossing_small_frame_delta(self):
        # 0.02 m per 100 ms; no 0.25m condition.
        g=route_xy(np.linspace(31.45,31.05,21),np.full(21,21.),4000)
        vel,_=kinematics(g)
        ev=crossings(g,PHY['E'],VIR['E'],SIGNALS,7,'E','W',vel,None,DEFAULTS)
        self.assertEqual(len(ev),1)
        self.assertEqual(ev[0]['event_type'],'red_light_running_candidate')

    def test_green_crossing(self):
        events,*_=crossing_ew(time0=15000)
        self.assertEqual(events[0]['event_type'],'green_crossing')

    def test_no_crossing_in_wrong_direction(self):
        g=route_xy(np.linspace(29.7,32.6,30),np.full(30,21),5000)
        vel,_=kinematics(g)
        ev=crossings(g,PHY['E'],VIR['E'],SIGNALS,7,'E','W',vel,None,DEFAULTS)
        self.assertEqual(len(ev),0)

    def test_green_then_wait_red_not_violation(self):
        # E/W turns green 13.68s; passes at ~15s, stops at 43s when red.
        x=np.r_[np.linspace(33,30,40), np.full(25,30.0)]
        t=np.r_[np.linspace(14200,16200,40),np.linspace(43000,46000,25)]
        g=pd.DataFrame(dict(track_id=['WAIT']*len(t),agent_type=['car']*len(t),timestamp_ms=t,x=x,y=np.full(len(t),21.)))
        vel,speed=kinematics(g)
        cc=crossings(g,PHY['E'],VIR['E'],SIGNALS,7,'E','W',vel,None,DEFAULTS)
        self.assertEqual(len(cc),1)
        self.assertEqual(cc[0]['event_type'],'green_crossing')
        stop=wait_events(g,PHY['E'],VIR['E'],SIGNALS,7,'E','W',vel,speed,None,DEFAULTS,cc)
        self.assertTrue(any(e['stop_position']=='past_stopline' for e in stop))
        self.assertTrue(all(e['interpretation']=='observed_waiting_not_independent_violation' for e in stop))

    def test_wrongway_outgoing_lane(self):
        # W outgoing road lane R3Out: x decreases from junction, correct direction west.
        # Drive eastwards -> wrongway.
        g=route_xy(np.linspace(-20,-5,55),np.full(55,20.4),5000)
        vel,speed=kinematics(g)
        ev=wrong_way_events(g,ROADTREE,vel,speed,DEFAULTS)
        self.assertGreaterEqual(len(ev),1)

    def test_correct_outgoing_lane_not_wrongway(self):
        g=route_xy(np.linspace(-5,-20,55),np.full(55,20.4),5000)
        vel,speed=kinematics(g)
        ev=wrong_way_events(g,ROADTREE,vel,speed,DEFAULTS)
        self.assertEqual(ev,[])

    def test_transverse_lane_crossing_not_wrongway(self):
        # Cross W outward lane perpendicular to its legal x-axis, no against flow.
        g=route_xy(np.full(60,-10),np.linspace(8,24,60),5000)
        vel,speed=kinematics(g)
        self.assertEqual(wrong_way_events(g,ROADTREE,vel,speed,DEFAULTS),[])

    def test_geographic_origin_spookdriver(self):
        # S outgoing lanes are still in S geographic arm, no required legal flow.
        g=route_xy(np.full(25,12.4),np.linspace(-9,-2,25),5000)
        arm,share,_=endpoint_arm(g,ROADTREE,VIR,'start',DEFAULTS)
        self.assertEqual(arm,'S')

    def test_incomplete_center_start_not_imputed(self):
        # starts in middle of junction, then exits W; origin should not be W.
        g=route_xy(np.linspace(10,-15,80),np.full(80,19.9),5000)
        arm,share,_=endpoint_arm(g,ROADTREE,VIR,'start',DEFAULTS)
        self.assertEqual(arm,'')

    def test_recorded_replay_cases_from_previous_export(self):
        # Independent replay observations (not fed into detection).
        # The file is an excerpt from v3 trajectory-point CSV, NOT original metadata.
        d=pd.read_csv(HERE/'replay_controlepunten_UIT_V3_EXPORT.csv')
        expectations={4:('E','S','red_light_running_candidate'),
                      9:('E','N','right_turn_on_red'),
                      10:('E','N','right_turn_on_red'),
                      11:('E','N','right_turn_on_red'),
                      12:('E','W','green_crossing'),
                      17:('S','W','red_crossing_near_phase_transition')}
        for tid,(origin,dest,label) in expectations.items():
            g=d[d.track_id==tid].reset_index(drop=True)
            self.assertGreater(len(g),20)
            o,_,_=endpoint_arm(g,ROADTREE,VIR,'start',DEFAULTS)
            de,_,_=endpoint_arm(g,ROADTREE,VIR,'end',DEFAULTS)
            self.assertEqual((o,de),(origin,dest),f'OD track {tid}')
            vel,_=kinematics(g)
            events=crossings(g,PHY[o],VIR[o],SIGNALS,{'E':7,'S':5}[o],o,de,vel,None,DEFAULTS)
            self.assertTrue(any(ev['event_type']==label for ev in events),(tid,[e['event_type'] for e in events]))

    def test_unknown_initial_arm_not_forced(self):
        d=pd.read_csv(HERE/'replay_controlepunten_UIT_V3_EXPORT.csv')
        for tid in (1,2):
            g=d[d.track_id==tid].reset_index(drop=True)
            o,_,_=endpoint_arm(g,ROADTREE,VIR,'start',DEFAULTS)
            self.assertEqual(o,'',tid)

    def test_osm_lanelet_consistency(self):
        for arm in ARMS:
            incomings=[z for z in ROADS if z.arm==arm and z.flow=='in']
            self.assertGreaterEqual(len(incomings),2)

if __name__=='__main__':unittest.main()
