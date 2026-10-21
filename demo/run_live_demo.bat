@echo off
rem Live demo of the brain-age audit. Double-click, wait ~30 s, the browser opens at http://127.0.0.1:7860
rem Add --share at the end of the next line for a temporary public link.
"D:\brainage-env\Scripts\python.exe" "%~dp0live_demo.py" --project-dir "D:\ML_Project" --sfcn-repo "D:\sfcn_repo" --checkpoint "G:\My Drive\BrainAge_Project\checkpoints\sfcn_finetune_official\best.pt" %*
pause
