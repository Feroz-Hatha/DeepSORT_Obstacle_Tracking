import numpy as np
from trackers.kalman_filter import KalmanBoxTracker
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
              (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh + 1e-6)
    return o

class Sort:
    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3, appearance_weight=0.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_weight = appearance_weight  # weight for feature similarity
        self.trackers = []

    def update(self, dets=np.empty((0, 6)), features=None):
        """
        dets: np.array([[x1,y1,x2,y2,score,class_id], ...])
        features: list of feature vectors, one per detection
        """
        trks = np.zeros((len(self.trackers), 5))
        to_del = []

        for t, trk in enumerate(self.trackers):
            trk.predict()
            trks[t][:4] = trk.get_state()
            trks[t][4] = 0

        if len(trks) == 0 or len(dets) == 0:
            matches = np.empty((0, 2), dtype=int)
            unmatched_dets = np.arange(len(dets))
            unmatched_trks = np.arange(len(trks))
        else:
            cost_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)

            for d, det in enumerate(dets):
                for t, trk in enumerate(trks):
                    iou_score = iou(det[:4], trk[:4])
                    if self.trackers[t].feature is not None and features is not None:
                        appearance_dist = cosine(features[d], self.trackers[t].feature)
                        appearance_dist = np.clip(appearance_dist, 0, 1)  # range [0,1]
                    else:
                        appearance_dist = 1.0  # max distance if no feature

                    # Combine distances
                    cost = self.appearance_weight * appearance_dist + (1 - self.appearance_weight) * (1 - iou_score)
                    cost_matrix[d, t] = cost

            matched_indices = linear_sum_assignment(cost_matrix)
            matched_indices = np.asarray(matched_indices).T

            unmatched_dets, unmatched_trks = [], []

            for d in range(len(dets)):
                if d not in matched_indices[:, 0]:
                    unmatched_dets.append(d)
            for t in range(len(trks)):
                if t not in matched_indices[:, 1]:
                    unmatched_trks.append(t)

            matches = []
            for m in matched_indices:
                d, t = m
                iou_score = iou(dets[d, :4], trks[t, :4])
                if iou_score < self.iou_threshold:
                    unmatched_dets.append(d)
                    unmatched_trks.append(t)
                else:
                    matches.append(m)

            if len(matches) == 0:
                matches = np.empty((0, 2), dtype=int)
            else:
                matches = np.array(matches)

        # Update matched trackers
        for m in matches:
            d, t = m
            self.trackers[t].update(dets[d, :4], feature=features[d])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4], class_id=int(dets[i, 5]), feature=features[i])
            self.trackers.append(trk)

        # Remove dead trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        results = []
        for trk in self.trackers:
            if (trk.hits >= self.min_hits) or (trk.time_since_update <= self.max_age):
                d = trk.get_state()
                results.append(np.concatenate((d, [trk.id, trk.class_id])).reshape(1, -1))

        if len(results) > 0:
            return np.concatenate(results)
        return np.empty((0, 6))