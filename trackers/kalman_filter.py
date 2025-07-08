import numpy as np

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox, class_id=None, feature=None):
        """
        bbox: [x1, y1, x2, y2]
        class_id: COCO class
        feature: appearance feature vector (e.g., from ResNet)
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1] + 1e-6)

        self.kf = self.create_kalman_filter()
        self.kf['state'] = np.array([cx, cy, s, r, 0, 0, 0], dtype=float)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

        self.class_id = class_id
        self.feature = feature  # store initial appearance feature

    def create_kalman_filter(self):
        kf = {}
        kf['state'] = np.zeros(7)
        kf['P'] = np.eye(7)
        kf['F'] = np.eye(7)
        kf['Q'] = np.eye(7) * 0.01
        kf['H'] = np.eye(4, 7)
        kf['R'] = np.eye(4) * 0.1
        return kf

    def predict(self):
        F = np.eye(7)
        F[0, 4] = 1
        F[1, 5] = 1
        F[2, 6] = 1

        self.kf['state'] = np.dot(F, self.kf['state'])
        self.kf['P'] = np.dot(F, np.dot(self.kf['P'], F.T)) + self.kf['Q']
        self.age += 1
        self.time_since_update += 1
        return self.kf['state']

    def update(self, bbox, feature=None):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1] + 1e-6)

        z = np.array([cx, cy, s, r])

        H = self.kf['H']
        x = self.kf['state']
        P = self.kf['P']
        R = self.kf['R']

        y = z - np.dot(H, x)
        S = np.dot(H, np.dot(P, H.T)) + R
        K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))

        self.kf['state'] = x + np.dot(K, y)
        self.kf['P'] = np.dot(np.eye(7) - np.dot(K, H), P)

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        if feature is not None:
            self.feature = feature  # update stored appearance feature

    def get_state(self):
        cx, cy, s, r = self.kf['state'][:4]
        s = max(s, 1e-6)
        r = max(r, 1e-6)
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return [x1, y1, x2, y2]