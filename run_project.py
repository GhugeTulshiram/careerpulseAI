from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

print("1/3 Cleaning data...")
subprocess.run([sys.executable, str(ROOT/"src"/"clean_data.py")], check=True)

print("2/3 Training ML models...")
subprocess.run([sys.executable, str(ROOT/"src"/"train_models.py")], check=True)

print("3/3 Creating database...")
subprocess.run([sys.executable, str(ROOT/"src"/"database.py")], check=True)

print("\nCareerPulse AI is ready.")
print("Run the dashboard with:")
print("streamlit run dashboard/app.py")
