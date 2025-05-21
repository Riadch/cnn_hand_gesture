import pygame
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import pandas as pd
import time
from threading import Thread
from queue import Queue


class HandGestureCNN(nn.Module):
    def __init__(self, num_classes=6): 
        super(HandGestureCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.fc_input_size = 512 * 4 * 4
        
    
        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        
        x = x.view(-1, self.fc_input_size)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
class GestureDetector:
    def __init__(self, model_path, classes_path):
        self.load_classes(classes_path)
        
        self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.current_gesture = None
        self.confidence = 0.0
        self.running = False
        self.frame = None
        self.gesture_queue = Queue(maxsize=1)
    
    def load_classes(self, classes_path):
        """Charge les noms des classes de gestes"""
        try:
            if os.path.exists(classes_path):
                classes_df = pd.read_csv(classes_path)
                self.classes = classes_df['class_name'].tolist()
                
            else:
                self.classes = [
                    "Gesture_0", "Gesture_11", "Gesture_2", 
                    "Gesture_5", "Gesture_6", "Gesture_9"
                ]
        
            print("Classes chargées:", self.classes)
        except Exception as e:
            print(f"Erreur lors du chargement des classes: {str(e)}")
            sys.exit(1)
    
    def load_model(self, model_path):
        """Charge le modèle de reconnaissance de gestes"""
        try:
            if os.path.exists(model_path):
                
                self.model = HandGestureCNN(num_classes=len(self.classes))
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                print("Modèle chargé avec succès!")
            else:
                print(f"Modèle non trouvé: {model_path}")
                sys.exit(1)
            
            self.model.eval()
            
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {str(e)}")
            sys.exit(1)
    
    def preprocess_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        img_tensor = self.transform(frame_rgb)
        
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    def detect_gesture(self, frame):
        img_tensor = self.preprocess_frame(frame)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            gesture_idx = torch.argmax(probabilities).item()
            confidence = probabilities[gesture_idx].item()
        
        self.current_gesture = self.classes[gesture_idx]
        self.confidence = confidence
        
        return self.current_gesture, self.confidence
    
    def start(self):
        self.running = True
        self.detection_thread = Thread(target=self.run_detection)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
    
    def run_detection(self):
        while self.running:
            ret, frame = self.cap.read()
            
            if ret:
                frame = cv2.flip(frame, 1)
                
                gesture, confidence = self.detect_gesture(frame)
                
                try:
                    
                    if self.gesture_queue.full():
                        self.gesture_queue.get_nowait()
                    
                    self.gesture_queue.put((gesture, confidence, frame.copy()))
                except:
                    pass
            time.sleep(0.005)
    
    def get_current_gesture(self):
        try:
            if not self.gesture_queue.empty():
                gesture, confidence, frame = self.gesture_queue.get_nowait()
                self.frame = frame
                return gesture, confidence, frame
        except:
            pass
        
        return self.current_gesture, self.confidence, self.frame

class PuzzleGame:
    def __init__(self, gesture_detector):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Jeu de Puzzle avec Gestes")
        
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.gesture_detector = gesture_detector
        self.cell_size = 50
        self.grid_width = 10
        self.grid_height = 10
        self.grid_offset_x = (self.width - self.grid_width * self.cell_size) // 2
        self.grid_offset_y = (self.height - self.grid_height * self.cell_size) // 2
        
        self.grid = [[0 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        self.robot_x = 0
        self.robot_y = 0
        
        self.exit_x = self.grid_width - 1
        self.exit_y = self.grid_height - 1
        
        self.obstacles = []
        self.generate_obstacles(20)
        
        self.game_over = False
        self.win = False
        
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.move_delay = 1 
        self.last_move_time = 0
        
        self.gesture_actions = {
            "Gesture_0": "stop",      
            "Gesture_2": "up",        
            "Gesture_11": "down",      
            "Gesture_5": "left",     
            "Gesture_6": "right",     
            "Gesture_9": "restart"   
        }
        self.font = pygame.font.SysFont(None, 36)
        self.small_font = pygame.font.SysFont(None, 24)
    
    def generate_obstacles(self, num_obstacles):
        import random
        
        self.obstacles = []
        
        safe_positions = [(self.robot_x, self.robot_y), (self.exit_x, self.exit_y)]
        
        
        for _ in range(num_obstacles):
            while True:
                x = random.randint(0, self.grid_width - 1)
                y = random.randint(0, self.grid_height - 1)
                
                if (x, y) not in safe_positions and (x, y) not in self.obstacles:
                    self.obstacles.append((x, y))
                    break
    
    def handle_gesture(self, gesture, confidence):
        if confidence < 0.96:
            return
        
        if gesture not in self.gesture_actions:
            return
        
        current_time = time.time()
        if current_time - self.last_move_time < self.move_delay:
            return
        
        self.last_move_time = current_time
        
        action = self.gesture_actions[gesture]
        
        if action == "restart":
            self.restart_game()
            return
        
        if self.game_over:
            return
        
        new_x, new_y = self.robot_x, self.robot_y
        
        if action == "up":
            new_y = max(0, self.robot_y - 1)
        elif action == "down":
            new_y = min(self.grid_height - 1, self.robot_y + 1)
        elif action == "left":
            new_x = max(0, self.robot_x - 1)
        elif action == "right":
            new_x = min(self.grid_width - 1, self.robot_x + 1)
        
        if (new_x, new_y) not in self.obstacles:
            self.robot_x, self.robot_y = new_x, new_y
        if self.robot_x == self.exit_x and self.robot_y == self.exit_y:
            self.win = True
            self.game_over = True
    
    def restart_game(self):
        self.robot_x = 0
        self.robot_y = 0
        self.generate_obstacles(20)
        self.game_over = False
        self.win = False
    
    def draw_grid(self):
        self.screen.fill(self.BLACK)
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, self.WHITE, rect, 1)
        
        for x, y in self.obstacles:
            rect = pygame.Rect(
                self.grid_offset_x + x * self.cell_size,
                self.grid_offset_y + y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, self.RED, rect)
        
        exit_rect = pygame.Rect(
            self.grid_offset_x + self.exit_x * self.cell_size,
            self.grid_offset_y + self.exit_y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.GREEN, exit_rect)
        
        robot_rect = pygame.Rect(
            self.grid_offset_x + self.robot_x * self.cell_size,
            self.grid_offset_y + self.robot_y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.BLUE, robot_rect)
    
    def draw_camera_feed(self, frame):
        if frame is None:
            return
        frame = cv2.resize(frame, (160, 120))
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        
        self.screen.blit(frame, (10, 10))
    
    def draw_gesture_info(self, gesture, confidence):
        gesture_text = f"Geste: {gesture}"
        gesture_surface = self.font.render(gesture_text, True, self.WHITE)
        self.screen.blit(gesture_surface, (10, 140))
        
        confidence_text = f"Confiance: {confidence*100:.1f}%"
        confidence_surface = self.font.render(confidence_text, True, self.WHITE)
        self.screen.blit(confidence_surface, (10, 180))

        if gesture in self.gesture_actions:
            action = self.gesture_actions[gesture]
            action_text = f"Action: {action}"
            action_surface = self.font.render(action_text, True, self.YELLOW)
            self.screen.blit(action_surface, (10, 220))
    
    def draw_instructions(self):
        instructions = [
            "Contrôles:",
            "(Gesture_0): Stop",
            "(Gesture_2): Haut",
            "(Gesture_11): Bas",
            "(Gesture_5): Gauche",
            "(Gesture_6): Droite",
            "(Gesture_9): Redémarrer"
        ]
        
        y_offset = 300
        for instruction in instructions:
            text_surface = self.small_font.render(instruction, True, self.WHITE)
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25
    
    def draw_game_over(self):
        if self.game_over:
            if self.win:
                text = "VICTOIRE! Pouce en haut pour recommencer"
                color = self.GREEN
            else:
                text = "PERDU! Pouce en haut pour recommencer"
                color = self.RED
            
            text_surface = self.font.render(text, True, color)
            text_rect = text_surface.get_rect(center=(self.width // 2, 50))
            self.screen.blit(text_surface, text_rect)
    
    def run(self):
        self.gesture_detector.start()
        
        try:
            
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r:
                            self.restart_game()
                gesture, confidence, frame = self.gesture_detector.get_current_gesture()
                
                if gesture:
                    self.handle_gesture(gesture, confidence)
                
                self.draw_grid()
                
                self.draw_camera_feed(frame)
                self.draw_gesture_info(gesture, confidence)
                
                self.draw_instructions()
                
                self.draw_game_over()
                
                pygame.display.flip()
                
                self.clock.tick(self.fps)
        
        finally:
            self.gesture_detector.stop()
            pygame.quit()

def main():
    model_path = './custom_cnn_game_best.pth'
    classes_path = './classes.csv'
    
   
    if not os.path.exists(model_path):
        print(f"Modèle non trouvé: {model_path}")
        print("Veuillez entraîner le modèle d'abord ou spécifier le chemin correct.")
        sys.exit(1)
    
    gesture_detector = GestureDetector(model_path, classes_path)
    
    game = PuzzleGame(gesture_detector)
    
    game.run()

if __name__ == "__main__":
    main()