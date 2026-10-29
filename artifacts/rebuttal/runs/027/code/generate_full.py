"""Use the unchanged generation engine on the full frozen inputs."""
from pathlib import Path
import generate_tasks

generate_tasks.RUN=Path(__file__).resolve().parents[1]/'full'

if __name__=='__main__':generate_tasks.main()
