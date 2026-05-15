export type Activity = 'WALKING' | 'WALKING_UPSTAIRS' | 'WALKING_DOWNSTAIRS' | 'SITTING' | 'STANDING' | 'LAYING';

export interface SessionRecord {
  id: string;
  date: string;
  activity: Activity;
  confidence: number;
  duration?: string;
  model: string;
}

export interface ModelResult {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  type: 'ML' | 'DL';
}

export interface PredictionResult {
  activity: Activity;
  confidence: number;
  probabilities: Record<Activity, number>;
}
