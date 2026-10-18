@echo off
echo ========================================================
echo  Generating 25 Synthetic Digits via Trained DBN...
echo ========================================================
python -m models.generate --samples 25 --gibbs_steps 100 --seed 42
pause
