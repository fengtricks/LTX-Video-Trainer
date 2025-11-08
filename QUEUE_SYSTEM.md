# Job Queue System for LTX-Video Trainer

The job queue system allows you to manage multiple training jobs efficiently. Jobs are stored in a persistent SQLite database and processed by a background worker.

## Features

✅ **Persistent Job Queue** - Jobs survive UI restarts  
✅ **Background Processing** - Worker runs independently from UI  
✅ **Job Management** - View, cancel, and resume jobs  
✅ **Auto-refresh** - Job list updates every 5 seconds  
✅ **Progress Tracking** - Real-time training progress and status  

## Quick Start

### Option 1: Start Everything Together (Recommended)

```bash
python scripts/start_with_queue.py
```

This starts both the job worker and Gradio UI in one command.

### Option 2: Start Separately

**Terminal 1 - Start the Worker:**
```bash
python scripts/jobs/run_worker.py
```

**Terminal 2 - Start the UI:**
```bash
python scripts/app_gradio.py
```

## Usage

### 1. Create and Queue a Training Job

1. Go to the **Datasets** tab
2. Create/upload/preprocess your dataset
3. Go to the **Training** tab
4. Configure training parameters
5. Click "▶️ Start Training"
6. Job is queued automatically!

### 2. Monitor Jobs

1. Go to the **Queue** tab
2. View all jobs in the table
3. Jobs auto-refresh every 5 seconds
4. Click on a job to see details

### 3. Manage Jobs

**Cancel a Job:**
- Select job from table
- Click "⏹️ Cancel Job"

**Resume a Failed/Cancelled Job:**
- Select the job
- Click "▶️ Resume Job"
- Creates a new job with same parameters

**Clear Completed Jobs:**
- Click "🗑️ Clear Completed"
- Removes all finished jobs from the list

## Job Status

- **pending** 🟡 - Waiting to be processed
- **running** 🔵 - Currently training
- **completed** 🟢 - Successfully finished
- **failed** 🔴 - Training encountered an error
- **cancelled** ⚪ - Stopped by user

## How It Works

### Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│             │         │              │         │             │
│  Gradio UI  │────────▶│   SQLite DB  │◀────────│   Worker    │
│             │  queue  │    jobs.db   │  poll   │             │
└─────────────┘  jobs   └──────────────┘  jobs   └─────────────┘
                                                         │
                                                         ▼
                                                  ┌─────────────┐
                                                  │  Training   │
                                                  │  Process    │
                                                  └─────────────┘
```

1. **UI** - Creates jobs and stores them in the database
2. **Database** - Persists jobs and their status
3. **Worker** - Polls database for pending jobs
4. **Training** - Worker spawns training processes

### Database Schema

```sql
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY,
    status TEXT,
    dataset_name TEXT,
    params TEXT (JSON),
    progress TEXT,
    error_message TEXT,
    created_at TEXT,
    started_at TEXT,
    completed_at TEXT,
    process_id INTEGER,
    output_dir TEXT,
    checkpoint_path TEXT,
    validation_sample TEXT
)
```

## Advanced Usage

### Check Worker Status

```bash
# Check if worker is running
ps aux | grep run_worker

# Check worker logs (if started separately)
python scripts/queue/run_worker.py 2>&1 | tee worker.log
```

### Direct CLI Training (Bypass Queue)

```bash
python scripts/train_cli.py \
  --dataset my_dataset \
  --steps 1000 \
  --learning-rate 2e-4 \
  --validation-prompt "a cake shaped like a car"
```

### Database Management

```python
from scripts.jobs.database import JobDatabase

# Connect to database
db = JobDatabase("jobs.db")

# Get all jobs
jobs = db.get_all_jobs()

# Get specific job
job = db.get_job(job_id)

# Clear old jobs
db.clear_completed_jobs()
```

## Troubleshooting

### Jobs Not Processing

1. **Check if worker is running:**
   ```bash
   ps aux | grep run_worker
   ```

2. **Start worker if not running:**
   ```bash
   python scripts/queue/run_worker.py
   ```

### Job Stuck in "Running"

If UI crashes while a job is running, it may remain marked as "running":

1. Cancel the stuck job
2. Resume it to create a new job
3. Worker will pick it up automatically

### Worker Dies Unexpectedly

- Check logs for errors
- Ensure enough disk space/memory
- Verify dataset is preprocessed correctly

### Multiple Jobs Training Simultaneously

By default, the worker processes one job at a time. To run multiple workers:

```bash
# Terminal 1
python scripts/queue/run_worker.py

# Terminal 2  
python scripts/queue/run_worker.py
```

⚠️ **Warning:** Multiple workers will compete for GPU resources!

## Files

```
scripts/
├── jobs/
│   ├── __init__.py         - Package initialization
│   ├── database.py         - JobDatabase class (SQLite)
│   ├── worker.py           - QueueWorker class
│   └── run_worker.py       - Worker entry point
├── train_cli.py            - CLI training script
├── start_with_queue.py     - Launch worker + UI
└── app_gradio.py           - Gradio UI (with Queue tab)

jobs.db                     - SQLite database (auto-created)
```

## Tips

💡 **Preprocess First** - Always preprocess datasets in the Datasets tab before training  
💡 **Monitor Progress** - Check the Queue tab during training for real-time updates  
💡 **Batch Training** - Queue multiple jobs with different parameters to train overnight  
💡 **Resume Failed Jobs** - Use the Resume button to retry failed trainings  
💡 **Clean Up** - Regularly clear completed jobs to keep the queue tidy  

## Example Workflow

```bash
# 1. Start the system
python scripts/start_with_queue.py

# 2. In browser (http://localhost:7860):
#    - Datasets tab: Create dataset, upload videos, caption, preprocess
#    - Training tab: Configure parameters, start training (job #1)
#    - Training tab: Change parameters, start training (job #2)
#    - Training tab: Change parameters, start training (job #3)
#    - Queue tab: Monitor all jobs

# 3. Jobs process automatically one by one
# 4. Download completed models from outputs/ directory
```

## Support

For issues or questions, check:
- Training logs in the Queue tab
- Output files in `outputs/lora_r{rank}_job{id}/`
- Worker console output

