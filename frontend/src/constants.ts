import { ModelResult } from './types';

export const LABELS: Record<number, string> = {
  0: 'WALKING',
  1: 'WALKING_UPSTAIRS',
  2: 'WALKING_DOWNSTAIRS',
  3: 'SITTING',
  4: 'STANDING',
  5: 'LAYING'
};

export const MODEL_STATS: ModelResult[] = [
  { model: 'Logistic Regression', accuracy: 0.96, precision: 0.95, recall: 0.96, f1: 0.96, type: 'ML' },
  { model: 'Random Forest', accuracy: 0.98, precision: 0.98, recall: 0.98, f1: 0.98, type: 'ML' },
  { model: 'LSTM', accuracy: 0.90, precision: 0.89, recall: 0.90, f1: 0.89, type: 'DL' },
  { model: 'CNN + LSTM', accuracy: 0.89, precision: 0.88, recall: 0.89, f1: 0.88, type: 'DL' }
];

export const ACTIVITY_DISTRIBUTION = [
  { activity: 'WALKING', count: 1226 },
  { activity: 'WALKING_UPSTAIRS', count: 1073 },
  { activity: 'WALKING_DOWNSTAIRS', count: 986 },
  { activity: 'SITTING', count: 1286 },
  { activity: 'STANDING', count: 1374 },
  { activity: 'LAYING', count: 1407 }
];

export const FEATURE_IMPORTANCE = [
  { feature: 'tGravityAcc-min()-X', importance: 0.15 },
  { feature: 'fBodyAccJerk-entropy()-X', importance: 0.12 },
  { feature: 'tGravityAcc-mean()-Y', importance: 0.10 },
  { feature: 'angle(Y,gravityMean)', importance: 0.08 },
  { feature: 'tGravityAcc-max()-X', importance: 0.07 },
  { feature: 'fBodyAcc-mad()-X', importance: 0.06 },
  { feature: 'tBodyAcc-energy()-X', importance: 0.05 },
  { feature: 'tGravityAcc-energy()-Y', importance: 0.04 },
  { feature: 'fBodyAcc-max()-X', importance: 0.03 },
  { feature: 'fBodyAcc-energy()-X', importance: 0.02 }
].sort((a, b) => b.importance - a.importance);

export const TRAINING_CURVES = [
  { epoch: 1, loss: 0.8, val_loss: 0.85, acc: 0.6, val_acc: 0.58 },
  { epoch: 2, loss: 0.5, val_loss: 0.55, acc: 0.75, val_acc: 0.72 },
  { epoch: 3, loss: 0.35, val_loss: 0.42, acc: 0.85, val_acc: 0.80 },
  { epoch: 4, loss: 0.28, val_loss: 0.35, acc: 0.88, val_acc: 0.85 },
  { epoch: 5, loss: 0.22, val_loss: 0.30, acc: 0.90, val_acc: 0.87 },
  { epoch: 6, loss: 0.18, val_loss: 0.28, acc: 0.92, val_acc: 0.88 },
  { epoch: 7, loss: 0.15, val_loss: 0.25, acc: 0.93, val_acc: 0.89 },
  { epoch: 8, loss: 0.12, val_loss: 0.24, acc: 0.94, val_acc: 0.90 },
  { epoch: 9, loss: 0.10, val_loss: 0.23, acc: 0.95, val_acc: 0.90 },
  { epoch: 10, loss: 0.09, val_loss: 0.22, acc: 0.96, val_acc: 0.90 },
];

export const SESSION_HISTORY = [
  { id: '1', date: '2024-04-24 08:30', activity: 'WALKING' as any, confidence: 0.98, duration: '25:00', model: 'Random Forest' },
  { id: '2', date: '2024-04-24 10:15', activity: 'STANDING' as any, confidence: 0.94, duration: '12:45', model: 'LSTM' },
  { id: '3', date: '2024-04-23 18:20', activity: 'LAYING' as any, confidence: 0.99, duration: '08:20', model: 'Random Forest' },
  { id: '4', date: '2024-04-23 07:45', activity: 'WALKING_UPSTAIRS' as any, confidence: 0.88, duration: '05:30', model: 'CNN + LSTM' },
];
