import sys
import os
import cv2
import numpy as np
import torch
import random
import math
import winsound
import threading
import time
import pygame
import traceback

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget, 
                             QFrame, QSizePolicy,
                             QStackedLayout, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPalette, QBrush, QIcon

from mmpose.apis import MMPoseInferencer

def get_asset_path(filename):
    """Safely finds files whether running as script or EXE"""
    if hasattr(sys, '_MEIPASS'):
        # Running as EXE: files are in a temp folder
        base_dir = os.path.join(sys._MEIPASS, "pics")
    else:
        # Running as script: files are in local 'pics' folder
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics")
    
    return os.path.join(base_dir, filename).replace("\\", "/")

# ==========================================
#               CONFIGURATION
# ==========================================
WEBCAM_ID = 0          
CAP_BACKEND = cv2.CAP_ANY  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAM_WIDTH  = 1280
CAM_HEIGHT = 720

CV_COLOR_RED = (0, 0, 255)
CV_COLOR_GREEN = (0, 255, 0)
CV_COLOR_YELLOW = (0, 255, 255)
CV_COLOR_GREY = (100, 100, 100)

# --- RULES ---
STRICT_KNEE_THRESHOLD = 0.95
MAX_KNEE_ANGLE = 170.0       
MAX_TRUNK_LEAN = 25.0        
CALIB_DURATION = 3.0         
MOVEMENT_THRESHOLD = 10.0
ZONE_HOLD_DURATION = 1.0     
MIN_TORSO_RATIO = 0.80       
HIP_DROP_RATIO = 0.2         

# KEYPOINTS
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

SKELETON_LINKS = [
    (5,6), (5,11), (6,12), (11,12),
    (11,13), (13,15), (12,14), (14,16),
    (5,7), (7,9), (6,8), (8,10)
]

# ==========================================
#            AUDIO MANAGER
# ==========================================

class AudioManager:
    _SOUNDS = {}

    @classmethod
    def get_sound(cls, filename):
        if filename not in cls._SOUNDS:
            try:
                path = get_asset_path(filename)
                
                if os.path.exists(path):
                    cls._SOUNDS[filename] = pygame.mixer.Sound(path)
                else:
                    cls._SOUNDS[filename] = None
            except Exception as e:
                print(f"Error loading sound {filename}: {e}")
                cls._SOUNDS[filename] = None
        
        return cls._SOUNDS[filename]

    @classmethod
    def play(cls, type):
        if type == "CLICK":
            sound = cls.get_sound("sound.wav")
            if sound: sound.play()
        elif type == "TIME":
            sound = cls.get_sound("time.wav") 
            if sound: sound.play()
        elif type == "LEVEL":
            sound = cls.get_sound("level.wav") 
            if sound: sound.play()
        elif type == "EXIT":
            sound = cls.get_sound("exit.wav")
            if sound: sound.play()
        elif type == "SUCCESS":
            sound = cls.get_sound("success.wav")
            if sound: sound.play()
            else: cls.beep_async(1000, 150)
        elif type == "HIT":
            sound = cls.get_sound("hit.wav")
            if sound: sound.play()
            else: cls.beep_async(300, 300)
        elif type == "WARNING":
            sound = cls.get_sound("warning.wav")
            if sound: sound.play()
            else: cls.beep_async(600, 300)
        elif type == "TICK":
            sound = cls.get_sound("tick.wav")
            if sound: sound.play()
            else: cls.beep_async(800, 50)

    @staticmethod
    def beep_async(frequency, duration):
        try:
            threading.Thread(target=winsound.Beep, args=(frequency, duration), daemon=True).start()
        except: pass

# ==========================================
#            GAME LOGIC ENGINE
# ==========================================
class GameEngine:
    def __init__(self):
        self.mode = None 
        self.difficulty = 1
        self.duration = 180.0
        self.plane_speed = 10
        self.difficulty_offset = 0
        
        # STATE: WAITING_FOR_POSE -> COUNTDOWN -> GAME -> PAUSED
        self.phase = "WAITING_FOR_POSE" 
        
        self.score_dodged = 0
        self.score_total = 0
        self.time_left = 0
        
        # Baselines
        self.base_shoulder_width = 0.0 
        self.base_body_height = 0.0    
        self.base_torso_length = 0.0   
        self.base_hip_height = 0.0     
        
        # Live Tracking
        self.smooth_shoulder_width = 0.0
        self.smooth_ankle_y = 0.0
        self.alpha = 0.35 
        
        self.standing_leg_len = 0.0 
        self.standing_head_y = 0.0
        
        # Plane Logic
        self.plane_pos = -50
        self.plane_active = False
        self.plane_scored = False
        self.flight_altitude = 0
        self.is_altitude_locked = False 
        self.virtual_standing_hip_y = 0.0 
        self.current_scale = 1.0
        
        self.target_zone = "CENTER"
        self.current_zone = "UNKNOWN"
        self.zone_hold_timer = 0.0 
        self.frame_counter = 0
        
        self.feedback_text = ""
        self.feedback_color = CV_COLOR_YELLOW
        self.message_timer = 0.0 
        
        # --- TIMERS ---
        self.calib_timer = 0.0      
        self.stable_timer = 0.0     
        self.prev_nose_y = 0.0

        # Load Plane
        self.plane_img = None
        try:
            # USE NEW HELPER HERE
            img_path = get_asset_path("plane.png")
            
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) 
            if img is not None:
                self.plane_img = cv2.resize(img, (100, 80))
        except Exception as e:
            print(f"Error loading plane image: {e}")

    def reset(self):
        self.phase = "WAITING_FOR_POSE"
        self.score_dodged = 0
        self.score_total = 0
        self.time_left = self.duration
        
        # --- NEW: PLANE LIST MANAGEMENT ---
        self.planes = []  # List of dicts: {'x': float, 'y': float, 'scored': bool}
        self.spawn_timer = 0.0 # To control spawn rate if needed
        self.calibrated_spawn_y = 0.0
        # ----------------------------------
        self.locked_user_center = None 

        self.calib_timer = 0.0
        self.stable_timer = 0.0
        self.message_timer = 0.0
        self.prev_nose_y = 0.0
        self.zone_hold_timer = 0.0
        self.is_altitude_locked = False
        self.smooth_shoulder_width = 0.0
        self.smooth_ankle_y = 0.0
        
        if self.difficulty == 1:
            self.plane_speed = 12; self.difficulty_offset = 40
        elif self.difficulty == 2:
            self.plane_speed = 18; self.difficulty_offset = 85
        elif self.difficulty == 3:
            self.plane_speed = 25; self.difficulty_offset = 125
    # --- CALCULATIONS ---
    def get_stable_head_y(self, kps):
        points = [kps[i][1] for i in range(5) if kps[i][1] > 0]
        if not points: return kps[NOSE][1]
        return sum(points) / len(points)

    def calculate_trunk_angle(self, kps):
        s_x = (kps[LEFT_SHOULDER][0] + kps[RIGHT_SHOULDER][0]) / 2
        s_y = (kps[LEFT_SHOULDER][1] + kps[RIGHT_SHOULDER][1]) / 2
        h_x = (kps[LEFT_HIP][0] + kps[RIGHT_HIP][0]) / 2
        h_y = (kps[LEFT_HIP][1] + kps[RIGHT_HIP][1]) / 2
        dx = abs(s_x - h_x); dy = abs(s_y - h_y)
        if dy == 0: return 90.0
        return math.degrees(math.atan(dx/dy))

    def get_knee_angle(self, kps, side="LEFT"):
        if side == "LEFT": h, k, a = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        else: h, k, a = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        p1 = np.array(kps[h][:2]); p2 = np.array(kps[k][:2]); p3 = np.array(kps[a][:2]) 
        a_side = np.linalg.norm(p2 - p3); b_side = np.linalg.norm(p1 - p3); c_side = np.linalg.norm(p1 - p2) 
        if 2 * a_side * c_side == 0: return 180.0
        cosine_angle = (a_side**2 + c_side**2 - b_side**2) / (2 * a_side * c_side)
        cosine_angle = max(-1.0, min(1.0, cosine_angle))
        return math.degrees(math.acos(cosine_angle))

    def get_leg_compression_ratio(self, kps):
        l_dx = kps[LEFT_HIP][0] - kps[LEFT_ANKLE][0]; l_dy = kps[LEFT_HIP][1] - kps[LEFT_ANKLE][1]; l_len = math.hypot(l_dx, l_dy)
        r_dx = kps[RIGHT_HIP][0] - kps[RIGHT_ANKLE][0]; r_dy = kps[RIGHT_HIP][1] - kps[RIGHT_ANKLE][1]; r_len = math.hypot(r_dx, r_dy)
        current_len = (l_len + r_len) / 2.0
        if self.standing_leg_len == 0: return 1.0
        return current_len / self.standing_leg_len

    # --- UPDATE LOOP ---
    def update(self, frame, all_kps_list, dt):
        h, w = frame.shape[:2]
        self.frame_counter += 1
        
        if self.message_timer > 0:
            self.message_timer -= dt
            if self.message_timer <= 0: self.feedback_text = "" 

        # 1. DEFINE CALIBRATION BOX (Unified Logic)
        if self.mode == "SIDEWAYS" and self.phase == "GAME":
            current_margin = int(w * 0.25) #narrow for Gameplay
        else:
            current_margin = int(w * 0.35) # Narrow (Center) for Calibration & Stationary

        box_x1 = current_margin
        box_y1 = int(h * 0.1)
        box_x2 = w - current_margin
        box_y2 = h - 50 

        # 2. STRICT SELECTION (With Continuity & Elbows)
        kps = None 
        found_valid_person = False
        
        if all_kps_list is not None and len(all_kps_list) > 0:
            # Size approx = Distance between shoulders (Point 5 and 6)
            all_kps_list.sort(key=lambda p: math.hypot(p[5][0]-p[6][0], p[5][1]-p[6][1]), reverse=True)

            for person_kps in all_kps_list:
                # [ADDED ELBOWS 7,8] to make sideways detection stricter!
                body_indices = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
                
                # A. BOUNDARY CHECK
                all_parts_in = True
                for idx in body_indices:
                     # Skip elbows/wrists if they are not detected (score=0) to avoid false pauses
                     if idx in [7, 8] and (person_kps[idx][0] == 0 or person_kps[idx][1] == 0):
                         continue 

                     px, py = person_kps[idx]
                     if not (box_x1 < px < box_x2 and box_y1 < py < box_y2): 
                         all_parts_in = False; break
                
                if all_parts_in:
                    # B. CONTINUITY CHECK (Anti-Teleport)
                    # If we tracked someone before, ensure the new detection is close by
                    if self.phase == "GAME" and self.locked_user_center is not None:
                        current_hip_x = (person_kps[LEFT_HIP][0] + person_kps[RIGHT_HIP][0]) / 2
                        current_hip_y = (person_kps[LEFT_HIP][1] + person_kps[RIGHT_HIP][1]) / 2
                        
                        dist = math.hypot(current_hip_x - self.locked_user_center[0], 
                                          current_hip_y - self.locked_user_center[1])
                        
                        # Strict Threshold: Center cannot jump > 200px instantly
                        if dist < 200:
                            kps = person_kps
                            self.locked_user_center = (current_hip_x, current_hip_y)
                            found_valid_person = True
                            break 
                    else:
                        # First acquisition (or re-calibrating)
                        kps = person_kps
                        cx = (person_kps[LEFT_HIP][0] + person_kps[RIGHT_HIP][0]) / 2
                        cy = (person_kps[LEFT_HIP][1] + person_kps[RIGHT_HIP][1]) / 2
                        self.locked_user_center = (cx, cy)
                        found_valid_person = True
                        break

        # If strict check failed (user stepped out), lose the lock
        if not found_valid_person:
            self.locked_user_center = None
            kps = None

        # Draw Skeleton (Visible ONLY during Calibration or Pause)
        if kps is not None and self.phase != "GAME":
            for p1, p2 in SKELETON_LINKS:
                pt1 = (int(kps[p1][0]), int(kps[p1][1])); pt2 = (int(kps[p2][0]), int(kps[p2][1]))
                cv2.line(frame, pt1, pt2, (0, 200, 0), 4)

        # 3. Game Metrics
        in_box = False; trunk_angle = 90.0; is_still = False
        if kps is not None:
            in_box = True 
            trunk_angle = self.calculate_trunk_angle(kps)
            nose_y = kps[NOSE][1]
            movement = abs(nose_y - self.prev_nose_y)
            self.prev_nose_y = nose_y
            is_still = movement < MOVEMENT_THRESHOLD

        # ===============================================
        #  PHASE 1: WAITING FOR POSE
        # ===============================================
        if self.phase == "WAITING_FOR_POSE":
            color = CV_COLOR_GREEN if in_box else CV_COLOR_YELLOW
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, 3)
            is_strict_straight = trunk_angle < 15.0 

            if not in_box: self.feedback_text = "STEP INSIDE BOX"; self.feedback_color = CV_COLOR_YELLOW; self.stable_timer = 0.0
            elif not is_strict_straight: self.feedback_text = "STAND STRAIGHT"; self.feedback_color = CV_COLOR_RED; self.stable_timer = 0.0
            elif not is_still: self.feedback_text = "STAND STILL"; self.feedback_color = CV_COLOR_RED; self.stable_timer = 0.0
            else:
                self.stable_timer += dt
                remaining = max(0.0, 2.0 - self.stable_timer)
                self.feedback_text = f"HOLD STEADY... {remaining:.1f}"
                self.feedback_color = CV_COLOR_GREEN
                if self.stable_timer >= 2.0: self.phase = "COUNTDOWN"; self.calib_timer = 0.0; AudioManager.play("TICK")

        # ===============================================
        #  PHASE 2: COUNTDOWN (4s Timer)
        # ===============================================
        elif self.phase == "COUNTDOWN":
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), CV_COLOR_GREEN, 3)
            is_strict_straight = trunk_angle < 15.0
            if not in_box or not is_strict_straight or not is_still:
                self.phase = "WAITING_FOR_POSE"; self.stable_timer = 0.0; AudioManager.play("WARNING"); return 

            self.calib_timer += dt
            remaining = max(0, CALIB_DURATION - self.calib_timer)
            display_num = int(math.ceil(remaining))
            self.feedback_text = f"CALIBRATING: {display_num}"; self.feedback_color = CV_COLOR_GREEN
            if int(remaining) != int(remaining + dt): AudioManager.play("TICK")
            
            if self.calib_timer >= CALIB_DURATION:
                # 1. Capture Standard Metrics
                l_dx = kps[LEFT_HIP][0] - kps[LEFT_ANKLE][0]; l_dy = kps[LEFT_HIP][1] - kps[LEFT_ANKLE][1]; l_len = math.hypot(l_dx, l_dy)
                r_dx = kps[RIGHT_HIP][0] - kps[RIGHT_ANKLE][0]; r_dy = kps[RIGHT_HIP][1] - kps[RIGHT_ANKLE][1]; r_len = math.hypot(r_dx, r_dy)
                self.standing_leg_len = (l_len + r_len) / 2.0
                self.standing_head_y = self.get_stable_head_y(kps)
                
                sx = kps[LEFT_SHOULDER][0] - kps[RIGHT_SHOULDER][0]; sy = kps[LEFT_SHOULDER][1] - kps[RIGHT_SHOULDER][1]
                self.base_shoulder_width = math.hypot(sx, sy)
                
                avg_ankle_y = (kps[LEFT_ANKLE][1] + kps[RIGHT_ANKLE][1]) / 2.0
                self.base_body_height = avg_ankle_y - kps[NOSE][1] # Height in pixels
                
                s_mid_y = (kps[LEFT_SHOULDER][1] + kps[RIGHT_SHOULDER][1]) / 2.0
                h_mid_y = (kps[LEFT_HIP][1] + kps[RIGHT_HIP][1]) / 2.0
                self.base_torso_length = abs(h_mid_y - s_mid_y)
                
                avg_hip_y = (kps[LEFT_HIP][1] + kps[RIGHT_HIP][1]) / 2.0
                self.base_hip_height = avg_ankle_y - avg_hip_y
                
                self.smooth_shoulder_width = self.base_shoulder_width
                self.smooth_ankle_y = avg_ankle_y
                
                # 2. CALCULATE DYNAMIC HEIGHT (Perspective Fix)
                # We assume a "Standard Person" is ~500px tall on screen.
                scale_factor = self.base_body_height / 500.0
                
                # Apply scale to the difficulty offset
                # max(0.5, ...) ensures it doesn't get too tiny if you are extremely far
                dynamic_offset = self.difficulty_offset * max(0.5, scale_factor)
                
                # Set the spawn height
                self.calibrated_spawn_y = self.standing_head_y + dynamic_offset
                
                # 3. RESET GAME STATE
                self.planes = [] 
                self.plane_active = False; self.plane_pos = -50; self.plane_scored = False; self.is_altitude_locked = False
                self.zone_hold_timer = 0.0; self.phase = "GAME"
                self.feedback_text = ""
                if self.mode == "SIDEWAYS": self.target_zone = "CENTER"

        # ===============================================
        #  PHASE 3: GAMEPLAY
        # ===============================================
        elif self.phase == "GAME":
            if not in_box: self.phase = "PAUSED_OUT_OF_BOUNDS"; AudioManager.play("WARNING"); return

            self.time_left -= dt
            if self.time_left <= 0: self.phase = "GAMEOVER"; AudioManager.play("WARNING"); return

            if kps is not None and len(kps) > 16:
                if not self.is_altitude_locked:
                    curr_sx = kps[LEFT_SHOULDER][0] - kps[RIGHT_SHOULDER][0]; curr_sy = kps[LEFT_SHOULDER][1] - kps[RIGHT_SHOULDER][1]; raw_width = math.hypot(curr_sx, curr_sy)
                    raw_ankle_y = (kps[LEFT_ANKLE][1] + kps[RIGHT_ANKLE][1]) / 2.0
                    if self.smooth_shoulder_width == 0: self.smooth_shoulder_width = raw_width
                    if self.smooth_ankle_y == 0: self.smooth_ankle_y = raw_ankle_y
                    self.smooth_shoulder_width = (raw_width * self.alpha) + (self.smooth_shoulder_width * (1.0 - self.alpha))
                    self.smooth_ankle_y = (raw_ankle_y * self.alpha) + (self.smooth_ankle_y * (1.0 - self.alpha))
                    if self.base_shoulder_width > 0: raw_scale = self.smooth_shoulder_width / self.base_shoulder_width
                    else: raw_scale = 1.0
                    if self.mode == "STATIONARY": self.current_scale = raw_scale
                    elif self.mode == "SIDEWAYS" and self.current_scale == 0: self.current_scale = 1.0
                    projected_height = self.base_body_height * self.current_scale
                    virtual_nose_y = self.smooth_ankle_y - projected_height
                    dynamic_offset = self.difficulty_offset * self.current_scale
                    projected_hip_h = self.base_hip_height * self.current_scale
                    self.virtual_standing_hip_y = self.smooth_ankle_y - projected_hip_h
                    self.flight_altitude = virtual_nose_y + dynamic_offset

                # --- STATIONARY MODE ---
                if self.mode == "STATIONARY":
                    should_spawn = False
                    if len(self.planes) == 0: should_spawn = True
                    elif self.planes[-1]['scored']:
                        if self.planes[-1].get('scored_time', 0.0) >= 0.5:
                            if not self.planes[-1].get('has_spawned_next', False):
                                should_spawn = True; self.planes[-1]['has_spawned_next'] = True

                    if should_spawn:
                        self.planes.append({'x': -50, 'y': self.calibrated_spawn_y, 'scored': False, 'scored_time': 0.0, 'has_spawned_next': False})

                    for plane in self.planes[:]:
                        plane['x'] += self.plane_speed
                        if plane['scored']: plane['scored_time'] += dt
                        self.flight_altitude = plane['y'] 
                        if w//2 - 50 < plane['x'] < w//2 + 50:
                            if not plane['scored']: self.resolve_collision_strict(kps); plane['scored'] = True 
                        if plane['x'] > w:
                            if not plane['scored']: self.score_total += 1; self.trigger_message("MISSED!", CV_COLOR_RED); plane['scored'] = True 
                            self.planes.remove(plane)
                    self.draw_plane(frame)

                # --- SIDEWAYS MODE (UPDATED) ---
                elif self.mode == "SIDEWAYS":
                    # 1. ZONES: Increased Center to 15% width
                    # Left: 0-42.5% | Center: 42.5-57.5% | Right: 57.5-100%
                    x1, x2 = int(w*0.425), int(w*0.575)
                    
                    cv2.line(frame, (x1,0), (x1,h), CV_COLOR_GREY, 2)
                    cv2.line(frame, (x2,0), (x2,h), CV_COLOR_GREY, 2)
                    
                    sh_mid = ((kps[LEFT_SHOULDER][0] + kps[RIGHT_SHOULDER][0]) / 2.0, (kps[LEFT_SHOULDER][1] + kps[RIGHT_SHOULDER][1]) / 2.0)
                    sx = int(sh_mid[0])
                    
                    if sx < x1: self.current_zone = "LEFT"
                    elif sx > x2: self.current_zone = "RIGHT"
                    else: self.current_zone = "CENTER"
                    
                    # 2. GAME FLOW LOGIC
                    should_spawn = False
                    
                    # We are ready for the next instruction if:
                    # A. There are no planes OR
                    # B. The last plane has already been hit/scored
                    ready_for_next = (len(self.planes) == 0) or (self.planes[-1]['scored'] == True)

                    if ready_for_next:
                        # LOGIC: Wait 0.5s after hit before showing "MOVE TO..." text
                        show_instructions = True
                        if len(self.planes) > 0 and self.planes[-1]['scored']:
                            if self.planes[-1]['scored_time'] < 0.5:
                                show_instructions = False

                        if show_instructions:
                            in_correct_zone = (self.current_zone == self.target_zone)
                            if in_correct_zone:
                                is_loose_straight = trunk_angle < 50.0 
                                if not is_loose_straight: self.feedback_text = "STAND STRAIGHT"; self.feedback_color = CV_COLOR_RED; self.zone_hold_timer = 0.0
                                elif not is_still: self.feedback_text = "STAND STILL"; self.feedback_color = CV_COLOR_RED; self.zone_hold_timer = 0.0
                                else:
                                    self.zone_hold_timer += dt; remaining = max(0.0, ZONE_HOLD_DURATION - self.zone_hold_timer)
                                    display_num = int(math.ceil(remaining)); self.feedback_text = f"HOLD: {display_num}"; self.feedback_color = CV_COLOR_GREEN
                                    if int(remaining) != int(remaining + dt): AudioManager.play("TICK")
                                    
                                    # HOLD COMPLETE -> SPAWN
                                    if self.zone_hold_timer >= ZONE_HOLD_DURATION: 
                                        should_spawn = True; self.zone_hold_timer = 0.0; self.feedback_text = ""
                                        if self.target_zone == "CENTER":
                                            curr_sx = kps[LEFT_SHOULDER][0] - kps[RIGHT_SHOULDER][0]; curr_sy = kps[LEFT_SHOULDER][1] - kps[RIGHT_SHOULDER][1]
                                            current_width = math.hypot(curr_sx, curr_sy)
                                            if self.base_shoulder_width > 0: self.current_scale = current_width / self.base_shoulder_width
                            else:
                                # User needs to move
                                self.zone_hold_timer = 0.0
                                if self.message_timer <= 0: self.feedback_text = f"MOVE TO {self.target_zone}!"; self.feedback_color = CV_COLOR_YELLOW
                            
                            self.draw_zone_highlight(frame, w, h, x1, x2)

                    # 3. SPAWN LOGIC
                    if should_spawn:
                        # Prevent double spawning from the same "ready" event
                        can_add = True
                        if len(self.planes) > 0:
                            if self.planes[-1].get('has_spawned_next', False): can_add = False
                            else: self.planes[-1]['has_spawned_next'] = True
                        
                        if can_add:
                            self.planes.append({'x': -50, 'y': self.calibrated_spawn_y, 'scored': False, 'scored_time': 0.0, 'has_spawned_next': False})

                    # 4. UPDATE & COLLISION
                    for plane in self.planes[:]:
                        plane['x'] += self.plane_speed
                        if plane['scored']: plane['scored_time'] += dt
                        
                        self.flight_altitude = plane['y'] 
                        
                        # Collision Check (Tracking User Position 'sx')
                        if sx - 100 < plane['x'] < sx + 100:
                            if not plane['scored']: 
                                self.resolve_collision_strict(kps)
                                plane['scored'] = True 
                                # TRIGGER NEW TARGET IMMEDIATELY ON HIT
                                self.pick_new_target() 

                        # Off-screen Check
                        if plane['x'] > w:
                            if not plane['scored']: 
                                self.score_total += 1; self.trigger_message("MISSED!", CV_COLOR_RED); plane['scored'] = True 
                                self.pick_new_target() # Also pick new target on miss
                            
                            self.planes.remove(plane)
                            
                    self.draw_plane(frame)

        # ===============================================
        #  PHASE 4: PAUSED (OUT OF BOUNDS)
        # ===============================================
        elif self.phase == "PAUSED_OUT_OF_BOUNDS":
            # Force Narrow Box for re-calibration
            narrow_margin = int(w * 0.35)
            box_x1 = narrow_margin; box_y1 = int(h * 0.1); box_x2 = w - narrow_margin; box_y2 = h - 50 
            
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), CV_COLOR_YELLOW, 3)
            self.feedback_text = "OUT OF BOUNDS! RECALIBRATE"
            self.feedback_color = CV_COLOR_RED
            sub_text = "STEP INSIDE BOX TO RESUME"
            t_size = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_TRIPLEX, 0.8, 2)[0]; t_x = (w - t_size[0]) // 2
            self.draw_shadow_text(frame, sub_text, (t_x, 100), 0.8, CV_COLOR_YELLOW)
            
            # Check strict re-entry
            # (Re-use strict boundary check here to confirm user is back in narrow box)
            if kps is not None:
                all_parts_in = True
                for idx in [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]:
                     if idx in [7, 8] and (kps[idx][0] == 0 or kps[idx][1] == 0): continue
                     px, py = kps[idx]
                     if not (box_x1 < px < box_x2 and box_y1 < py < box_y2): all_parts_in = False; break
                
                if all_parts_in:
                    self.phase = "WAITING_FOR_POSE"; self.stable_timer = 0.0; self.feedback_text = "HOLD STEADY..."
        
        # FINAL STEP: DRAW GLOBAL FEEDBACK TEXT
        if self.feedback_text:
            font_scale = 1.2; thickness = 3
            text_size = cv2.getTextSize(self.feedback_text, cv2.FONT_HERSHEY_TRIPLEX, font_scale, thickness)[0]
            text_w, text_h = text_size; center_x = (w - text_w) // 2; center_y = 60 
            cv2.putText(frame, self.feedback_text, (center_x + 2, center_y + 2), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 0), thickness)
            cv2.putText(frame, self.feedback_text, (center_x, center_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, self.feedback_color, thickness)

    # --- HELPER: DRAW ZONES (Missing Function) ---
    def draw_zone_highlight(self, frame, w, h, x1, x2):
        # Flashing effect every 10 frames
        if (self.frame_counter // 10) % 2 == 0:
            overlay = frame.copy()
            if self.target_zone == "LEFT": 
                cv2.rectangle(overlay, (0, 0), (x1, h), CV_COLOR_YELLOW, -1)
            elif self.target_zone == "RIGHT": 
                cv2.rectangle(overlay, (x2, 0), (w, h), CV_COLOR_YELLOW, -1)
            else: # CENTER
                cv2.rectangle(overlay, (x1, 0), (x2, h), CV_COLOR_YELLOW, -1)
            
            # Apply transparent overlay (alpha blending)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    def pick_new_target(self):
        # 1. Define all possible targets
        options = ["LEFT", "CENTER", "RIGHT"]
        
        # 2. Remove the zone we are currently in (so we don't get the same one twice)
        if self.current_zone in options:
            options.remove(self.current_zone)
            
        # 3. Pick a random new target
        self.target_zone = random.choice(options)

    def draw_plane(self, frame):
        h, w = frame.shape[:2]
        
        for plane in self.planes:
            # --- NEW: VISIBILITY CHECK ---
            # If plane has been scored (hit/success) for > 0.5s, skip drawing it (Invisible)
            if plane.get('scored', False) and plane.get('scored_time', 0.0) >= 0.5:
                continue
            # -----------------------------

            ix, iy = int(plane['x']), int(plane['y'])
            
            # Draw Yellow Line
            cv2.line(frame, (0, iy), (w, iy), CV_COLOR_YELLOW, 2)

            if self.plane_img is None:
                continue

            ph, pw = self.plane_img.shape[:2]
            x_start = ix - pw // 2; y_start = iy - ph // 2; x_end = x_start + pw; y_end = y_start + ph
            x1 = max(0, x_start); y1 = max(0, y_start); x2 = min(w, x_end); y2 = min(h, y_end)
            
            if x1 < x2 and y1 < y2:
                px1 = x1 - x_start; py1 = y1 - y_start; px2 = px1 + (x2 - x1); py2 = py1 + (y2 - y1)
                roi = frame[y1:y2, x1:x2]
                plane_chunk = self.plane_img[py1:py2, px1:px2]
                if plane_chunk.shape[2] == 4:
                    plane_rgb = plane_chunk[:, :, :3]
                    alpha_mask = plane_chunk[:, :, 3] / 255.0
                    alpha_inv = 1.0 - alpha_mask
                    for c in range(3):
                        roi[:, :, c] = (alpha_mask * plane_rgb[:, :, c] + alpha_inv * roi[:, :, c])
                    frame[y1:y2, x1:x2] = roi

    def resolve_collision_strict(self, kps):
        current_head_y = self.get_stable_head_y(kps); is_under_line = current_head_y > self.flight_altitude
        l_angle = self.get_knee_angle(kps, "LEFT"); r_angle = self.get_knee_angle(kps, "RIGHT"); avg_knee_angle = (l_angle + r_angle) / 2.0
        is_knees_physically_bent = avg_knee_angle < MAX_KNEE_ANGLE
        compression_ratio = self.get_leg_compression_ratio(kps); is_ratio_good = compression_ratio < STRICT_KNEE_THRESHOLD
        is_valid_squat = is_ratio_good and is_knees_physically_bent
        trunk_angle = self.calculate_trunk_angle(kps); is_upright = trunk_angle < MAX_TRUNK_LEAN
        s_mid_y = (kps[LEFT_SHOULDER][1] + kps[RIGHT_SHOULDER][1]) / 2.0; h_mid_y = (kps[LEFT_HIP][1] + kps[RIGHT_HIP][1]) / 2.0; current_torso_len = abs(h_mid_y - s_mid_y)
        is_torso_valid = current_torso_len > (self.base_torso_length * MIN_TORSO_RATIO)
        current_hip_y = (kps[LEFT_HIP][1] + kps[RIGHT_HIP][1]) / 2.0; hip_drop = current_hip_y - self.virtual_standing_hip_y
        required_drop = self.difficulty_offset * self.current_scale * HIP_DROP_RATIO; is_hip_dropped = hip_drop > required_drop
        self.score_total += 1; self.plane_scored = True 
        if not is_under_line: self.trigger_message("HIT! DUCK LOWER", CV_COLOR_RED); AudioManager.play("HIT")
        else:
            if not is_valid_squat: self.trigger_message("HIT! BEND YOUR KNEES", CV_COLOR_RED); AudioManager.play("HIT")
            elif not is_upright: self.trigger_message("HIT! KEEP BACK STRAIGHT", CV_COLOR_RED); AudioManager.play("HIT")
            elif not is_torso_valid and not is_hip_dropped: self.trigger_message("HIT! DON'T LEAN FORWARD", CV_COLOR_RED); AudioManager.play("WARNING")
            else: self.score_dodged += 1; self.trigger_message("PERFECT DODGE!", CV_COLOR_GREEN); AudioManager.play("SUCCESS")

    def trigger_message(self, text, color): self.feedback_text = text; self.feedback_color = color; self.message_timer = 2.0
    
    def draw_shadow_text(self, img, text, pos, scale, color):
        cv2.putText(img, text, (pos[0]+2, pos[1]+2), cv2.FONT_HERSHEY_TRIPLEX, scale, (0,0,0), 2)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_TRIPLEX, scale, color, 2)

# ==========================================
#            WORKER THREAD (VIDEO) - PERSISTENT
# ==========================================
class VideoWorker(QThread):
    frame_update = pyqtSignal(QImage)
    stats_update = pyqtSignal(int, int, str, int, str)
    game_over = pyqtSignal()
    camera_error = pyqtSignal(str) 

    def __init__(self, game_engine, inferencer):
        super().__init__()
        self.game = game_engine
        self.inferencer = inferencer
        self.running = True
        self.is_game_active = False # Flag to control processing

    # --- OPTIMIZED RUN LOOP (Reduces Lag) ---
    def run(self):
        cap = None
        try:
            if os.name == 'nt':
                cap = cv2.VideoCapture(WEBCAM_ID, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(WEBCAM_ID, CAP_BACKEND)
            
            # Resolution - Lowering slightly can also help lag if needed
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            
            time.sleep(1.0)

            if not cap.isOpened():
                self.camera_error.emit(f"CAMERA {WEBCAM_ID} FAILED")
                self.running = False
                return

            last_time = time.time()
            frame_skip_counter = 0  # Counter for skipping
            current_kps = None      # Store last known keypoints

            while self.running:
                ret, frame = cap.read()
                if not ret: 
                    time.sleep(0.01)
                    continue

                if self.is_game_active:
                    frame = cv2.flip(frame, 1)
                    current_time = time.time()
                    dt = current_time - last_time
                    last_time = current_time

                    # --- OPTIMIZATION: Process AI only every 2nd frame ---
                    if frame_skip_counter % 2 == 0:
                        result = self.inferencer(frame, show=False)
                        current_kps = self.extract_kps(result)
                    
                    frame_skip_counter += 1

                    # Update game logic (uses last known KPS if skipped)
                    self.game.update(frame, current_kps, dt)

                    if self.game.phase == "GAMEOVER":
                        self.game_over.emit()
                        self.is_game_active = False

                    #if self.game.feedback_text:
                     #   self.draw_shadow_text(frame, self.game.feedback_text, (CAM_WIDTH//2 - 200, 100), 1.2, self.game.feedback_color)

                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                    self.frame_update.emit(qt_image)
                    
                    color_hex = "#00FF00" if self.game.feedback_color == CV_COLOR_GREEN else "#FF0000" if self.game.feedback_color == CV_COLOR_RED else "#FFFF00"
                    self.stats_update.emit(self.game.score_dodged, self.game.score_total, self.game.feedback_text, int(self.game.time_left), color_hex)
                else:
                    time.sleep(0.03)

        except Exception as e:
            traceback.print_exc()
            self.camera_error.emit("CRITICAL THREAD ERROR")
        finally:
            if cap: cap.release()

    def extract_kps(self, result):
        try:
            result_list = list(result)
            # Get list of ALL instances (people) detected in the frame
            instances = result_list[0]["predictions"][0] 
            all_kps = []
            for inst in instances:
                all_kps.append(np.array(inst["keypoints"], dtype=float))
            return all_kps # Returns a LIST of skeletons
        except:
            return []

# ==========================================
#            PYQT USER INTERFACE
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        pygame.mixer.init(44100, -16, 2, 256) # Initialize audio mixer
        self.setWindowTitle("Aeroplane Game")
        self.setStyleSheet("QMainWindow { background-color: #1e1e2e; } QLabel { color: white; font-family: 'Segoe UI'; }")
        self.shared_inferencer = MMPoseInferencer(
            pose2d="human",
            device=DEVICE
            )
        self.game_engine = GameEngine()
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        # Initialize the persistent worker immediately
        self.worker = VideoWorker(self.game_engine, self.shared_inferencer)
        self.worker.frame_update.connect(self.update_video_frame)
        self.worker.stats_update.connect(self.update_hud)
        self.worker.game_over.connect(self.game_finished)
        self.worker.camera_error.connect(self.show_error_screen)
        self.worker.start()

        self.init_landing_screen() 
        self.init_instructions_screen()
        self.init_menu_screen() 
        self.init_settings_screen()
        self.init_game_screen()
        self.init_scorecard_screen() 

    # Override closeEvent to properly release camera on exit
    def closeEvent(self, event):
        if self.worker:
            self.worker.running = False
            self.worker.quit()
            self.worker.wait()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            # ONLY exit full screen, DO NOT stop the game logic
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

    # --- NEW: LANDING PAGE (HORIZONTAL BUTTONS) ---
    def init_landing_screen(self):
        page = QWidget()
        
        # Background
        palette = QPalette()
        screen_rect = QApplication.primaryScreen().size()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_bg = get_asset_path("bg1.png")

        if os.path.exists(path_bg):
            img = QImage(path_bg).scaled(screen_rect, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            palette.setBrush(QPalette.Window, QBrush(img))
            page.setAutoFillBackground(True)
            page.setPalette(palette)
        else:
            page.setStyleSheet("background-color: #0d1b2a;")

        # Layouts
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignBottom | Qt.AlignCenter)
        main_layout.setContentsMargins(0, 0, 0, 150)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(50)
        button_layout.setAlignment(Qt.AlignCenter)

        # STYLE: No ugly box. The button 'Floats Up' when hovered.
        LANDING_BTN_STYLE = """
            QPushButton {
                background-color: transparent;
                border: none;
                /* Default Position: Sits slightly lower */
                padding-top: 10px;
                padding-bottom: 0px;
            }
            QPushButton:hover {
                /* Hover Effect: Moves UP by 10 pixels */
                padding-top: 0px;
                padding-bottom: 10px;

            }
            QPushButton:pressed {
                /* Click Effect: Pushes DOWN */
                padding-top: 15px;
                padding-bottom: 0px;
                border-bottom: none;
            }
        """

        # Play Button
        btn_play = QPushButton()
        btn_play.setFixedSize(350, 140)
        btn_play.setStyleSheet(LANDING_BTN_STYLE)
        path_play = get_asset_path("play.png")
        if os.path.exists(path_play):
            btn_play.setIcon(QIcon(path_play))
            btn_play.setIconSize(QSize(330, 120))
        else:
            btn_play.setText("PLAY")

        # Exit Button
        btn_exit = QPushButton()
        btn_exit.setFixedSize(350, 140)
        btn_exit.setStyleSheet(LANDING_BTN_STYLE)
        path_exit = get_asset_path("exit.png")
        if os.path.exists(path_exit):
            btn_exit.setIcon(QIcon(path_exit))
            btn_exit.setIconSize(QSize(330, 120))
        else:
            btn_exit.setText("EXIT")

        # Logic
        btn_play.clicked.connect(lambda: AudioManager.play("CLICK"))
        btn_play.clicked.connect(lambda: self.central_widget.setCurrentIndex(1))
        
        btn_exit.clicked.connect(lambda: AudioManager.play("EXIT")) 
        btn_exit.clicked.connect(self.close)

        button_layout.addWidget(btn_play)
        button_layout.addWidget(btn_exit)
        main_layout.addLayout(button_layout)
        page.setLayout(main_layout)
        self.central_widget.addWidget(page)

    def init_menu_screen(self):
        page = QWidget()
        
        # 1. Background Image Setup
        palette = QPalette()
        screen_rect = QApplication.primaryScreen().size()
        bg_path = get_asset_path("bg2.png")
        
        if os.path.exists(bg_path):
            bg_image = QImage(bg_path).scaled(screen_rect, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            palette.setBrush(QPalette.Window, QBrush(bg_image))
            page.setAutoFillBackground(True)
            page.setPalette(palette)
        else:
            page.setStyleSheet("background-color: #1e1e2e;") 

        # 2. Main Layout (Centers the Container)
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        # 3. THE CONTAINER BOX
        menu_box = QFrame()
        menu_box.setFixedSize(600, 650) # Big enough to hold title and 3 buttons
        menu_box.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 46, 0.90); /* Dark Semi-Transparent BG */
                border: 4px solid #89b4fa; /* Blue Border */
                border-radius: 40px;
            }
        """)

        # Layout inside the box
        box_layout = QVBoxLayout(menu_box)
        box_layout.setAlignment(Qt.AlignCenter)
        box_layout.setSpacing(30) # Spacing between buttons

        # --- STYLES ---

        # Title Style (Transparent background, just text)
        TITLE_STYLE = """
            QLabel {
                border: none;
                background-color: transparent;
                font-family: 'Segoe UI', sans-serif;
                font-size: 48px;
                font-weight: 900;
                color: #89b4fa;
            }
        """

        # Mode Buttons Style (Blue Theme)
        BTN_MODE_STYLE = """
            QPushButton {
                background-color: rgba(49, 50, 68, 0.5); /* Slightly transparent inside box */
                color: #cdd6f4;
                border: 3px solid #89b4fa;
                border-radius: 20px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 24px;
                font-weight: 900;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #89b4fa; 
                color: #1e1e2e;
                border: 3px solid #b4befe;
            }
            QPushButton:pressed {
                background-color: #74c7ec;
            }
        """

        # Back Button Style (Pink Theme)
        BTN_BACK_STYLE = """
            QPushButton {
                background-color: rgba(49, 50, 68, 0.5);
                color: #f38ba8;
                border: 3px solid #f38ba8;
                border-radius: 20px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 24px;
                font-weight: 900;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #f38ba8;
                color: #1e1e2e;
            }
            QPushButton:pressed {
                background-color: #d97e9c;
            }
        """

        # --- UI ELEMENTS ---

        # Title
        title = QLabel("SELECT MODE")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(TITLE_STYLE)

        # Stationary Mode Button
        btn_stat = QPushButton("STATIONARY MODE")
        btn_stat.setFixedSize(450, 90) # Wider to fill box nicely
        btn_stat.setCursor(Qt.PointingHandCursor)
        btn_stat.setStyleSheet(BTN_MODE_STYLE)
        # Sound & Action
        btn_stat.clicked.connect(lambda: AudioManager.play("CLICK"))
        btn_stat.clicked.connect(lambda: self.go_to_settings("STATIONARY"))

        # Sideways Mode Button
        btn_side = QPushButton("SIDEWAYS MODE")
        btn_side.setFixedSize(450, 90)
        btn_side.setCursor(Qt.PointingHandCursor)
        btn_side.setStyleSheet(BTN_MODE_STYLE)
        # Sound & Action
        btn_side.clicked.connect(lambda: AudioManager.play("CLICK"))
        btn_side.clicked.connect(lambda: self.go_to_settings("SIDEWAYS"))

        # Back Button
        btn_exit = QPushButton("BACK")
        btn_exit.setFixedSize(300, 80)
        btn_exit.setCursor(Qt.PointingHandCursor)
        btn_exit.setStyleSheet(BTN_BACK_STYLE)
        # Sound & Action
        btn_exit.clicked.connect(lambda: AudioManager.play("EXIT"))
        btn_exit.clicked.connect(lambda: self.central_widget.setCurrentIndex(0)) 

        # --- ASSEMBLY ---
        box_layout.addStretch(1)
        box_layout.addWidget(title, 0, Qt.AlignCenter)
        box_layout.addSpacing(40)
        box_layout.addWidget(btn_stat, 0, Qt.AlignCenter)
        box_layout.addWidget(btn_side, 0, Qt.AlignCenter)
        box_layout.addSpacing(40)
        box_layout.addWidget(btn_exit, 0, Qt.AlignCenter)
        box_layout.addStretch(1)

        # Add the box to the main layout
        main_layout.addWidget(menu_box)
        
        page.setLayout(main_layout)
        self.central_widget.addWidget(page)
    # --- UPDATED SETTINGS SCREEN (Background + Button Styles) ---
    def init_settings_screen(self):
        self.settings_page = QWidget()

        palette = QPalette()
        screen_rect = QApplication.primaryScreen().size()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_bg_settings = get_asset_path("bg2.png")

        if os.path.exists(path_bg_settings):
            img = QImage(path_bg_settings).scaled(screen_rect, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            palette.setBrush(QPalette.Window, QBrush(img))
            self.settings_page.setAutoFillBackground(True)
            self.settings_page.setPalette(palette)
        else:
            self.settings_page.setStyleSheet("background-color: #1e1e2e;")

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(30)

        self.settings_page.setStyleSheet("""
            QPushButton[class="level_btn"] {
                background-color: rgba(49, 50, 68, 0.9); color: #cdd6f4; border: 2px solid #89b4fa;
                border-radius: 15px; font-family: 'Segoe UI', sans-serif; font-size: 20px; font-weight: 900; padding: 10px;
            }
            QPushButton[class="level_btn"]:hover { background-color: #89b4fa; color: #1e1e2e; border: 3px solid #b4befe; }
            QPushButton[class="level_btn"]:checked { background-color: #fab387; color: #1e1e2e; border: 3px solid #ffffff; }
            
            QPushButton[class="time_btn"] {
                background-color: #313244; color: #cdd6f4; border-radius: 10px; font-weight: bold; border: 2px solid #45475a; font-size: 16px;
            }
            QPushButton[class="time_btn"]:hover { background-color: #45475a; border-color: #89b4fa; color: white; }
            
            QLabel[class="header"] {
                font-size: 18px; font-weight: bold; color: #cdd6f4; background-color: rgba(30,30,46,0.6); padding: 5px 15px; border-radius: 8px;
            }
        """)

        self.lbl_mode = QLabel("MODE: STATIONARY")
        self.lbl_mode.setAlignment(Qt.AlignCenter)
        self.lbl_mode.setStyleSheet("font-size: 32px; font-weight: 900; color: #89b4fa; background-color: rgba(30,30,46,0.7); border-radius: 15px; padding: 10px 25px; border: 2px solid #89b4fa;")

        lbl_diff_title = QLabel("SELECT DIFFICULTY")
        lbl_diff_title.setProperty("class", "header")
        lbl_diff_title.setAlignment(Qt.AlignCenter)

        diff_layout = QHBoxLayout()
        diff_layout.setSpacing(25)
        diff_layout.setAlignment(Qt.AlignCenter)
        
        self.btn_easy = QPushButton("EASY (Lv 1)")
        self.btn_med = QPushButton("MEDIUM (Lv 2)")
        self.btn_hard = QPushButton("HARD (Lv 3)")
        
        for btn in [self.btn_easy, self.btn_med, self.btn_hard]:
            btn.setFixedSize(220, 70); btn.setCheckable(True)
            btn.setProperty("class", "level_btn"); btn.setCursor(Qt.PointingHandCursor)
            # --- SOUND ---
            btn.clicked.connect(lambda: AudioManager.play("LEVEL"))
            diff_layout.addWidget(btn)
        
        self.btn_easy.clicked.connect(lambda: self.set_difficulty(1))
        self.btn_med.clicked.connect(lambda: self.set_difficulty(2))
        self.btn_hard.clicked.connect(lambda: self.set_difficulty(3))

        lbl_time_title = QLabel("SESSION DURATION")
        lbl_time_title.setProperty("class", "header")
        lbl_time_title.setAlignment(Qt.AlignCenter)

        time_layout = QHBoxLayout()
        time_layout.setAlignment(Qt.AlignCenter)
        
        btn_minus = QPushButton("- 1 Min")
        btn_plus = QPushButton("+ 1 Min")
        btn_minus.setProperty("class", "time_btn")
        btn_plus.setProperty("class", "time_btn")
        
        self.lbl_time = QLabel("03:00")
        self.lbl_time.setStyleSheet("font-family: 'Arial'; font-size: 42px; font-weight: bold; color: #fab387; background-color: rgba(0,0,0,0.6); padding: 5px 30px; border-radius: 12px; border: 2px solid #fab387;")
        
        # --- SOUND ---
        btn_minus.clicked.connect(lambda: AudioManager.play("TIME"))
        btn_plus.clicked.connect(lambda: AudioManager.play("TIME"))

        btn_minus.clicked.connect(lambda: self.change_time(-60))
        btn_plus.clicked.connect(lambda: self.change_time(60))
        
        for btn in [btn_minus, btn_plus]:
             btn.setFixedSize(110, 55); btn.setCursor(Qt.PointingHandCursor)

        time_layout.addWidget(btn_minus); time_layout.addSpacing(25)
        time_layout.addWidget(self.lbl_time); time_layout.addSpacing(25)
        time_layout.addWidget(btn_plus)

        btn_start = QPushButton("START SESSION")
        btn_start.setFixedSize(400, 85); btn_start.setCursor(Qt.PointingHandCursor)
        # --- SOUND ---
        btn_start.clicked.connect(lambda: AudioManager.play("CLICK"))
        btn_start.clicked.connect(self.start_game)
        btn_start.setStyleSheet("""
            QPushButton { background-color: #a6e3a1; color: #1e1e2e; border: none; border-radius: 42px; font-size: 28px; font-weight: 900; }
            QPushButton:hover { background-color: #94e28d; border: 4px solid #ffffff; }
            QPushButton:pressed { background-color: #81c88b; }
        """)

        btn_back = QPushButton("BACK")
        btn_back.setFixedSize(160, 55); btn_back.setCursor(Qt.PointingHandCursor)
        # --- SOUND ---
        btn_back.clicked.connect(lambda: AudioManager.play("EXIT"))
        btn_back.clicked.connect(lambda: self.central_widget.setCurrentIndex(2))
        btn_back.setStyleSheet("""
            QPushButton { background-color: rgba(30, 30, 46, 0.8); color: #f38ba8; border: 3px solid #f38ba8; border-radius: 15px; font-weight: bold; font-size: 20px;}
            QPushButton:hover { background-color: #f38ba8; color: #1e1e2e; }
        """)

        layout.addStretch(1)
        layout.addWidget(self.lbl_mode); layout.addSpacing(30)
        layout.addWidget(lbl_diff_title); layout.addSpacing(10); layout.addLayout(diff_layout)
        layout.addSpacing(30)
        layout.addWidget(lbl_time_title); layout.addSpacing(10); layout.addLayout(time_layout)
        layout.addSpacing(50)
        layout.addWidget(btn_start, 0, Qt.AlignCenter); layout.addSpacing(20)
        layout.addWidget(btn_back, 0, Qt.AlignCenter)
        layout.addStretch(1)

        self.settings_page.setLayout(layout)
        self.central_widget.addWidget(self.settings_page)
        self.set_difficulty(1)

    # --- NEW: INSTRUCTIONS SCREEN (Scaled to Fit Box) ---
    def init_instructions_screen(self):
        self.instr_page = QWidget()

        # 1. Background Setup
        palette = QPalette()
        screen_rect = QApplication.primaryScreen().size()
        path_bg = get_asset_path("bg2.png")

        if os.path.exists(path_bg):
            img = QImage(path_bg).scaled(screen_rect, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            palette.setBrush(QPalette.Window, QBrush(img))
            self.instr_page.setAutoFillBackground(True)
            self.instr_page.setPalette(palette)
        else:
            self.instr_page.setStyleSheet("background-color: #1e1e2e;")

        # 2. Main Layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # 3. Content Container
        content_box = QFrame()
        content_box.setFixedSize(1150, 700) 
        content_box.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 46, 0.95); 
                border: 4px solid #89b4fa;
                border-radius: 40px;
            }
        """)
        
        box_layout = QVBoxLayout(content_box)
        box_layout.setSpacing(0)
        box_layout.setContentsMargins(15, 25, 15, 20) 

        # Title Label
        lbl_title = QLabel("HOW TO PLAY")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("""
            background-color: transparent;
            color: #89b4fa;
            font-family: 'Segoe UI';
            font-weight: 900;
            font-size: 65px;
            margin-bottom: 5px; 
            border: none;
        """)

        # Instructions Body Text
        lbl_instr = QLabel()
        lbl_instr.setWordWrap(True)
        lbl_instr.setTextFormat(Qt.RichText)
        lbl_instr.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lbl_instr.setStyleSheet("border: none; background-color: transparent;")
        
        # HTML Formatting (SCALED UP)
        instructions_html = """
        <div style='color: #cdd6f4; font-family: Segoe UI; font-size: 28px; line-height: 140%; text-align: left;'>
            <ul style='margin-left: 0px; padding-left: 10px; margin-top: 0px;'>
                <li style='margin-bottom: 20px;'>
                    <b style='color: #f9e2af;'>How to Win:</b> Perform perfect squats to avoid planes hitting you. Hits reduce your accuracy, so stay sharp!
                </li>
                <li style='margin-bottom: 20px;'>
                    <b style='color: #a6e3a1;'>Setup:</b> Align yourself within the <b>Green Box</b>. The game auto-calibrates when you stand still,<i style='color:#f38ba8;'>Do not change position excessively.</i>
                </li>
                <li style='margin-bottom: 20px;'>
                    <b>Stationary Mode:</b> Stand in place and <b>Squat</b> to avoid planes.
                </li>
                <li style='margin-bottom: 20px;'>
                    <b>Sideways Mode:</b> Agility test! Quickly move to the highlighted <b>Left/Right</b> zone before squatting.
                </li>
                <li style='margin-bottom: 20px;'>
                    <b>Perfect Form:</b> The camera tracks your body. Keep your chest up and bend your knees while squatting. Avoid excessive movement or improper posture.
                </li>
            </ul>
        </div>
        """
        lbl_instr.setText(instructions_html)

        # 5. Buttons (Start & Back)
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(50)
        btn_layout.setAlignment(Qt.AlignCenter)

        # BACK BUTTON
        btn_back = QPushButton("BACK")
        btn_back.setFixedSize(240, 75)
        btn_back.setCursor(Qt.PointingHandCursor)
        btn_back.clicked.connect(lambda: AudioManager.play("EXIT")) 
        btn_back.clicked.connect(lambda: self.central_widget.setCurrentIndex(0))
        btn_back.setStyleSheet("""
            QPushButton { background-color: rgba(30, 30, 46, 0.5); color: #f38ba8; border: 3px solid #f38ba8; border-radius: 20px; font-weight: bold; font-size: 22px;}
            QPushButton:hover { background-color: #f38ba8; color: #1e1e2e; }
        """)

        # START BUTTON
        btn_start = QPushButton("START")
        btn_start.setFixedSize(240, 75)
        btn_start.setCursor(Qt.PointingHandCursor)
        btn_start.clicked.connect(lambda: AudioManager.play("CLICK")) 
        btn_start.clicked.connect(lambda: self.central_widget.setCurrentIndex(2))
        btn_start.setStyleSheet("""
            QPushButton { background-color: #a6e3a1; color: #1e1e2e; border: none; border-radius: 20px; font-size: 24px; font-weight: 900; }
            QPushButton:hover { background-color: #94e28d; border: 4px solid #ffffff; }
            QPushButton:pressed { background-color: #81c88b; }
        """)

        btn_layout.addWidget(btn_back)
        btn_layout.addWidget(btn_start)

        # Assemble
        box_layout.addWidget(lbl_title)
        box_layout.addWidget(lbl_instr, 1) 
        box_layout.addLayout(btn_layout)
        
        layout.addWidget(content_box)
        self.instr_page.setLayout(layout)
        
        self.central_widget.addWidget(self.instr_page)
    # --- UPDATED GAME SCREEN (Grouped Info Boxes) ---
    def init_game_screen(self):
        page = QWidget()
        
        # Stack Layout: Video (0) -> Frame (1) -> UI (2)
        stack_layout = QStackedLayout()
        stack_layout.setStackingMode(QStackedLayout.StackAll)
        page.setLayout(stack_layout)

        # --- HELPER: PATH FINDER ---
        def get_path(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(base_dir, "pics", filename).replace("\\", "/")
            if os.path.exists(full_path):
                return full_path
            else:
                return None

        # --- STYLES ---
        BASE_STYLE = """
            background-color: rgba(20, 20, 30, 0.85);
            border-radius: 10px;
            font-family: 'Segoe UI', sans-serif;
            font-weight: 900;
            font-size: 18px;
            padding: 2px;
        """
        SCORE_STYLE = f"QLabel {{ {BASE_STYLE} border: 2px solid #89b4fa; color: white; }}"
        TIME_STYLE  = f"QLabel {{ {BASE_STYLE} border: 2px solid #f9e2af; color: #f9e2af; }}"

        BTN_RESTART_STYLE = f"""
            QPushButton {{ {BASE_STYLE} border: 2px solid #fab387; color: #fab387; }}
            QPushButton:hover {{ background-color: rgba(250, 179, 135, 0.2); border: 3px solid #fab387; color: white; }}
            QPushButton:pressed {{ background-color: #fab387; color: black; }}
        """
        BTN_EXIT_STYLE = f"""
            QPushButton {{ {BASE_STYLE} border: 2px solid #f38ba8; color: #f38ba8; }}
            QPushButton:hover {{ background-color: rgba(243, 139, 168, 0.2); border: 3px solid #f38ba8; color: white; }}
            QPushButton:pressed {{ background-color: #f38ba8; color: black; }}
        """

        # ===========================
        # LAYER 0 & 1: VIDEO & FRAME
        # ===========================
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("background-color: black;")
        
        
        # ===========================
        # LAYER 2: UI CONTROLS (Top Bar)
        # ===========================
        ui_wrapper = QWidget()
        ui_wrapper.setAttribute(Qt.WA_TranslucentBackground)
        main_ui_layout = QVBoxLayout(ui_wrapper)
        main_ui_layout.setContentsMargins(0, 0, 0, 0)

        # --- THE HEADER BAR ---
        header_widget = QWidget()
        header_widget.setAttribute(Qt.WA_TranslucentBackground)
        header_layout = QGridLayout(header_widget)
        # Margins: Left=30, Top=10, Right=30, Bottom=0
        header_layout.setContentsMargins(30, 10, 30, 0) 
        
        # --- 1. LEFT CONTAINER (Success + Time) ---
        info_container = QWidget()
        info_layout = QHBoxLayout(info_container)
        info_layout.setContentsMargins(0,0,0,0)
        info_layout.setSpacing(15) # Space between Success and Time

        # SUCCESS BOX
        self.lbl_score = QLabel("SUCCESS: 0 / 0")
        self.lbl_score.setFixedSize(200, 60)
        self.lbl_score.setAlignment(Qt.AlignCenter)
        self.lbl_score.setStyleSheet(SCORE_STYLE)

        # TIME BOX
        self.lbl_game_time = QLabel("TIME: 03:00")
        self.lbl_game_time.setFixedSize(200, 60)
        self.lbl_game_time.setAlignment(Qt.AlignCenter)
        self.lbl_game_time.setStyleSheet(TIME_STYLE)

        # Add to left container
        info_layout.addWidget(self.lbl_score)
        info_layout.addWidget(self.lbl_game_time)

        # --- 2. RIGHT CONTAINER (Buttons) ---
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(0,0,0,0)
        btn_layout.setSpacing(15)

        # RESTART BUTTON
        btn_restart = QPushButton("RESTART")
        btn_restart.setFixedSize(200, 60)
        btn_restart.setCursor(Qt.PointingHandCursor)
        btn_restart.setStyleSheet(BTN_RESTART_STYLE)
        btn_restart.clicked.connect(lambda: AudioManager.play("CLICK"))
        btn_restart.clicked.connect(self.restart_game_logic)

        # EXIT BUTTON
        btn_exit = QPushButton("EXIT")
        btn_exit.setFixedSize(200, 60)
        btn_exit.setCursor(Qt.PointingHandCursor)
        btn_exit.setStyleSheet(BTN_EXIT_STYLE)
        btn_exit.clicked.connect(lambda: AudioManager.play("EXIT"))
        btn_exit.clicked.connect(self.game_finished)

        # Add to right container
        btn_layout.addWidget(btn_restart)
        btn_layout.addWidget(btn_exit)

        # --- PLACE CONTAINERS IN GRID ---
        # (Widget, Row, Col, Alignment)
        
        # Left Group (Score + Time)
        header_layout.addWidget(info_container, 0, 0, Qt.AlignLeft | Qt.AlignTop)
        
        # Right Group (Restart + Exit)
        header_layout.addWidget(btn_container, 0, 1, Qt.AlignRight | Qt.AlignTop)

        # Add flexible space between them
        header_layout.setColumnStretch(0, 0) # Auto size left
        header_layout.setColumnStretch(1, 1) # Auto size right

        # Add Header to Main Layout
        main_ui_layout.addWidget(header_widget)
        main_ui_layout.addStretch(1) # Push everything to top

        # --- STACK LAYERS ---
        stack_layout.addWidget(self.video_label)   # 0
        stack_layout.addWidget(ui_wrapper)         # 2
        
        ui_wrapper.raise_()
        self.central_widget.addWidget(page)
        
    def init_scorecard_screen(self):
        self.score_page = QWidget()

        palette = QPalette()
        screen_rect = QApplication.primaryScreen().size()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_bg_score = get_asset_path("bg2.png")

        if os.path.exists(path_bg_score):
            img = QImage(path_bg_score).scaled(screen_rect, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            palette.setBrush(QPalette.Window, QBrush(img))
            self.score_page.setAutoFillBackground(True)
            self.score_page.setPalette(palette)
        else:
            self.score_page.setStyleSheet("background-color: #1e1e2e;")

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        score_card_box = QFrame()
        score_card_box.setFixedSize(700, 650) # Increased height slightly to fit new label
        score_card_box.setStyleSheet("""
            QFrame { background-color: rgba(30, 30, 46, 0.90); border: 4px solid #89b4fa; border-radius: 40px; }
        """)

        box_layout = QVBoxLayout(score_card_box); box_layout.setAlignment(Qt.AlignCenter); box_layout.setSpacing(15)

        lbl_title = QLabel("SESSION COMPLETE")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("border: none; background-color: transparent; font-family: 'Segoe UI', sans-serif; font-size: 48px; font-weight: 900; color: #89b4fa;")

        # DODGES (Green)
        self.lbl_final_dodges = QLabel("SUCCESSFUL DODGES: 0")
        self.lbl_final_dodges.setAlignment(Qt.AlignCenter)
        self.lbl_final_dodges.setStyleSheet("border: none; background-color: transparent; font-family: 'Segoe UI', sans-serif; font-size: 28px; font-weight: bold; color: #a6e3a1;")

        # HITS (Red)
        self.lbl_final_hits = QLabel("HITS / MISTAKES: 0")
        self.lbl_final_hits.setAlignment(Qt.AlignCenter)
        self.lbl_final_hits.setStyleSheet("border: none; background-color: transparent; font-family: 'Segoe UI', sans-serif; font-size: 28px; font-weight: bold; color: #f38ba8;")

        # --- NEW: ACCURACY (Gold/Yellow) ---
        self.lbl_final_accuracy = QLabel("ACCURACY: 0.0%")
        self.lbl_final_accuracy.setAlignment(Qt.AlignCenter)
        self.lbl_final_accuracy.setStyleSheet("border: none; background-color: transparent; font-family: 'Segoe UI', sans-serif; font-size: 32px; font-weight: 900; color: #f9e2af;")

        btn_exit_menu = QPushButton("EXIT TO MENU")
        btn_exit_menu.setFixedSize(350, 80); btn_exit_menu.setCursor(Qt.PointingHandCursor)
        
        # --- SOUND ---
        btn_exit_menu.clicked.connect(lambda: AudioManager.play("EXIT"))
        btn_exit_menu.clicked.connect(lambda: self.central_widget.setCurrentIndex(2))
        
        btn_exit_menu.setStyleSheet("""
            QPushButton {
                background-color: rgba(0,0,0,0.2); color: #f38ba8; border: 3px solid #f38ba8; 
                border-radius: 20px; font-family: 'Segoe UI', sans-serif; font-weight: 900; font-size: 24px;
            }
            QPushButton:hover { background-color: #f38ba8; color: #1e1e2e; }
            QPushButton:pressed { background-color: #d97e9c; border-color: #d97e9c; }
        """)

        box_layout.addStretch(1); box_layout.addWidget(lbl_title); box_layout.addSpacing(30)
        box_layout.addWidget(self.lbl_final_dodges)
        box_layout.addWidget(self.lbl_final_hits)
        box_layout.addSpacing(10)
        box_layout.addWidget(self.lbl_final_accuracy) # Added here
        box_layout.addSpacing(50); box_layout.addWidget(btn_exit_menu, 0, Qt.AlignCenter); box_layout.addStretch(1)

        main_layout.addWidget(score_card_box)
        self.score_page.setLayout(main_layout)
        self.central_widget.addWidget(self.score_page)

    def go_to_settings(self, mode):
        self.game_engine.mode = mode
        self.lbl_mode.setText(f"MODE: {mode}")
        self.central_widget.setCurrentIndex(3)

    def set_difficulty(self, level):
        self.game_engine.difficulty = level
        
        # Update UI state without destroying the Stylesheet
        self.btn_easy.setChecked(level == 1)
        self.btn_med.setChecked(level == 2)
        self.btn_hard.setChecked(level == 3)
    def change_time(self, seconds):
        new_time = self.game_engine.duration + seconds
        if 60 <= new_time <= 1500:
            self.game_engine.duration = new_time
            m = int(new_time // 60)
            s = int(new_time % 60)
            self.lbl_time.setText(f"{m:02d}:{s:02d}")

    def show_loading_screen(self):
        loading_pixmap = QPixmap(CAM_WIDTH, CAM_HEIGHT)
        loading_pixmap.fill(Qt.black)
        painter = QPainter(loading_pixmap)
        painter.setPen(QColor(255, 255, 0)) 
        font = QFont("Arial", 40, QFont.Bold)
        painter.setFont(font)
        painter.drawText(loading_pixmap.rect(), Qt.AlignCenter, "LOADING CAMERA...")
        painter.end()
        self.video_label.setPixmap(loading_pixmap)
        QApplication.processEvents()

    def show_error_screen(self, msg):
        error_pixmap = QPixmap(CAM_WIDTH, CAM_HEIGHT)
        error_pixmap.fill(Qt.black)
        painter = QPainter(error_pixmap)
        painter.setPen(QColor(255, 0, 0))
        font = QFont("Arial", 30, QFont.Bold)
        painter.setFont(font)
        painter.drawText(error_pixmap.rect(), Qt.AlignCenter, msg)
        painter.end()
        self.video_label.setPixmap(error_pixmap)

    def start_game(self):
        self.game_engine.reset()
        self.show_loading_screen()
        # Change 3 to 4, because Instructions is now 3
        self.central_widget.setCurrentIndex(4) 
        
        if self.worker:
            self.worker.is_game_active = True

    def restart_game_logic(self):
        self.game_engine.reset()
        self.lbl_score.setText("SUCCESS: 0 / 0")
        
    def stop_game(self):
        # UPDATED: Just pause the existing worker
        if self.worker:
            self.worker.is_game_active = False
        self.show_loading_screen()

    def game_finished(self):
        # 1. CAPTURE SCORES FIRST
        final_dodges = self.game_engine.score_dodged
        final_total = self.game_engine.score_total
        
        # 2. STOP THE GAME ENGINE
        self.stop_game()

        # 3. CALCULATE STATS
        hits = max(0, final_total - final_dodges)
        
        # Calculate Accuracy
        if final_total > 0:
            accuracy = (final_dodges / final_total) * 100.0
        else:
            accuracy = 0.0

        # 4. UPDATE UI
        self.lbl_final_dodges.setText(f"SUCCESSFUL DODGES: {final_dodges}")
        self.lbl_final_hits.setText(f"HITS / MISTAKES: {hits}")
        self.lbl_final_accuracy.setText(f"ACCURACY: {accuracy:.1f}%")
        
        # 5. FORCE REFRESH
        self.lbl_final_dodges.repaint()
        self.lbl_final_hits.repaint()
        self.lbl_final_accuracy.repaint()

        # 6. SWITCH SCREEN
        self.central_widget.setCurrentIndex(5) # Jump to Score (Index 5)

    def exit_to_menu(self):
        self.central_widget.setCurrentIndex(0) # Back to Landing (Index 0)

    def update_video_frame(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def update_hud(self, dodged, total, feedback, time_sec, color):
        # Updates the Score and Time text on the game screen
        if hasattr(self, 'lbl_score'):
            # CHANGED TO 'SUCCESS' HERE:
            self.lbl_score.setText(f"SUCCESS: {dodged} / {total}")
        
        if hasattr(self, 'lbl_game_time'):
            m = time_sec // 60
            s = time_sec % 60
            self.lbl_game_time.setText(f"TIME: {m:02d}:{s:02d}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showFullScreen()
    sys.exit(app.exec_())



