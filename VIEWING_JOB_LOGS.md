# 📝 How to View Job Logs

## Quick Guide

There are **TWO WAYS** to view training job logs in the Jobs tab:

### Method 1: Click on Job ID in Table (Recommended)

1. Go to the **Jobs** tab
2. Look at the job list table
3. **Click directly on the Job ID number** (first column)
4. The logs will appear in the "Training Logs" section below

**Important:** You must click on the **ID column** (the number), not the Status, Dataset, or other columns.

### Method 2: Manual Entry

1. Go to the **Jobs** tab
2. In the "Job Details" section on the right, you'll see a "Job ID" field
3. Type or paste the job ID number
4. Click the **"👁️ View Details"** button
5. The logs will appear in the "Training Logs" section below

## What You'll See

Once you select a job, you'll see:

- **Job Details** (JSON): Complete job information including status, timestamps, parameters
- **Training Logs** (Text): Real-time training output including:
  - Training progress (steps, epochs)
  - Loss values
  - Validation sample generation
  - Error messages (if job failed)

## Refreshing the Job List

Click the **"🔄 Refresh"** button to update the job list and see the latest status of all jobs.

## Troubleshooting

### "No logs available yet"
- The job hasn't started yet (still pending)
- The worker hasn't begun processing it

### "Click on a Job ID in the table to view details"
- You clicked on the wrong column (Status, Dataset, etc.)
- Solution: Click specifically on the **ID number** in the first column

### Empty logs section
- Make sure you've selected a job using one of the methods above
- Check that the "Training Logs" accordion is expanded

## Example

```
Jobs Table:
┌────┬──────────┬──────────────┬─────────────┬──────────┐
│ ID │ Status   │ Dataset      │ Created     │ Progress │
├────┼──────────┼──────────────┼─────────────┼──────────┤
│ 1  │ running  │ my_dataset   │ 2025-11-09  │ Step 50  │  <-- Click here!
│ 2  │ pending  │ other_data   │ 2025-11-09  │          │
└────┴──────────┴──────────────┴─────────────┴──────────┘
```

When you click on "1", the logs will load automatically.

## Tips

- The **Training Logs** section is now open by default for easier access
- Use the **copy button** (top-right of logs) to copy logs for sharing or debugging
- You can scroll through long log files
- Logs update in real-time as the job progresses

