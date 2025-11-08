#!/usr/bin/env python3
"""
Start both the queue worker and Gradio UI together.

This script launches:
1. Queue worker in the background to process training jobs
2. Gradio UI for user interaction

Both processes will be managed together, and stopping this script will stop both.
"""

import signal
import subprocess
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ServiceManager:
    """Manage worker and UI processes."""

    def __init__(self):
        self.worker_process = None
        self.ui_process = None
        self.running = True

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        print("\n🛑 Shutting down services...")
        self.stop()
        sys.exit(0)

    def start(self):
        """Start both worker and UI processes."""
        python_exe = sys.executable
        scripts_dir = Path(__file__).parent

        print("🚀 Starting LTX-Video Trainer with Queue System")
        print("=" * 60)

        # Start worker process
        print("\n📋 Starting job worker...")
        worker_script = scripts_dir / "jobs" / "run_worker.py"
        self.worker_process = subprocess.Popen(
            [python_exe, str(worker_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Give worker time to start
        time.sleep(2)

        if self.worker_process.poll() is not None:
            print("❌ Worker failed to start")
            return False

        print("✅ Queue worker started")

        # Start UI process
        print("\n🎨 Starting Gradio UI...")
        ui_script = scripts_dir / "app_gradio.py"
        self.ui_process = subprocess.Popen(
            [python_exe, str(ui_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Wait for UI to start
        time.sleep(3)

        if self.ui_process.poll() is not None:
            print("❌ UI failed to start")
            self.stop()
            return False

        print("✅ Gradio UI started")
        print("\n" + "=" * 60)
        print("🎉 All services running!")
        print("📍 Open your browser to: http://localhost:7860")
        print("💡 Press Ctrl+C to stop all services")
        print("=" * 60 + "\n")

        return True

    def monitor(self):
        """Monitor both processes and restart if needed."""
        while self.running:
            # Check if processes are still running
            if self.worker_process and self.worker_process.poll() is not None:
                print("⚠️  Worker process stopped unexpectedly")
                self.running = False
                break

            if self.ui_process and self.ui_process.poll() is not None:
                print("⚠️  UI process stopped unexpectedly")
                self.running = False
                break

            # Print output from processes
            if self.ui_process and self.ui_process.stdout:
                line = self.ui_process.stdout.readline()
                if line:
                    print(f"[UI] {line}", end="")

            time.sleep(0.1)

    def stop(self):
        """Stop both processes."""
        self.running = False

        if self.ui_process:
            print("Stopping UI...")
            self.ui_process.terminate()
            try:
                self.ui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ui_process.kill()

        if self.worker_process:
            print("Stopping worker...")
            self.worker_process.terminate()
            try:
                self.worker_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.worker_process.kill()

        print("✅ All services stopped")


def main():
    """Main entry point."""
    manager = ServiceManager()

    if not manager.start():
        print("❌ Failed to start services")
        sys.exit(1)

    try:
        manager.monitor()
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal")
    finally:
        manager.stop()


if __name__ == "__main__":
    main()

