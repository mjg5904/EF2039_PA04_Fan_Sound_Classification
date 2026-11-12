"""Classify a new stereo or mono WAV with a trained fan-condition model."""
import argparse
import joblib
import soundfile as sf
from run_experiments import audio_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audio', required=True)
    p.add_argument('--model', default='results/real/near_and_far_model.joblib')
    p.add_argument('--channel', choices=['near', 'far'], default='near')
    args = p.parse_args()
    model = joblib.load(args.model)
    audio, sr = sf.read(args.audio, always_2d=True, dtype='float32')
    channel = 0 if args.channel == 'near' else 1
    if channel >= audio.shape[1]:
        raise ValueError('Audio does not contain the selected channel')
    features = audio_features(audio[:, channel], sr).reshape(1, -1)
    prediction = model.predict(features)[0]
    print('Predicted fan condition:', prediction)
    if hasattr(model, 'predict_proba'):
        for label, prob in zip(model.classes_, model.predict_proba(features)[0]):
            print(f'  {label}: {prob:.3f}')


if __name__ == '__main__':
    main()
