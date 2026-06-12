
import os
root = '/root/autodl-tmp/IEMOCAP/wavs'
sessions = set()
for d in ['ANGER','HAPPY','NEUTRAL','SAD']:
    dp = os.path.join(root, d)
    if os.path.isdir(dp):
        for f in os.listdir(dp):
            if f.endswith('.wav'):
                sessions.add(f[:6])
print('Sessions:', sorted(sessions))
print('Count:', len(sessions))
