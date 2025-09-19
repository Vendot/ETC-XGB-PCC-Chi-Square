# ETCXGB Feature Selection Project - Docker Setup

This project implements a hybrid Extra Trees Classifier (ETC) and XGBoost model with feature selection for heart disease prediction.

## Docker Setup

### Prerequisites
- Docker
- Docker Compose

### Quick Start

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Access Jupyter Notebook:**
   Open your browser and go to: `http://localhost:8888`

3. **Stop the containers:**
   ```bash
   docker-compose down
   ```

### Alternative Docker Commands

1. **Build the Docker image:**
   ```bash
   docker build -t etcxgb-fs .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8888:8888 -v $(pwd):/app etcxgb-fs
   ```

### Development Mode

To run a persistent Python container for development:
```bash
docker-compose --profile dev up -d python
docker-compose exec python bash
```

### Project Structure
```
.
├── dataset.csv           # Heart disease dataset
├── ETCXGB_FS.ipynb      # Main notebook with ETC-XGB hybrid model
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker image configuration
├── docker-compose.yml   # Docker Compose configuration
├── .dockerignore        # Files to exclude from Docker build
└── README.md           # This file
```

### Features
- **Feature Selection**: PCC (Pearson Correlation Coefficient) and Chi-Square methods
- **Hybrid Model**: Extra Trees Classifier feeding into XGBoost
- **Cross Validation**: Stratified K-Fold validation
- **Multiple Scenarios**: Baseline, PCC, Chi2, and combined approaches

### Usage

1. Start the Docker environment
2. Open the Jupyter notebook interface at `http://localhost:8888`
3. Run the `ETCXGB_FS.ipynb` notebook
4. Experiment with different feature selection parameters:
   - `pcc_threshold`: Threshold for PCC feature selection (default: 0.2)
   - `chi2_k`: Number of features to select with Chi-Square (default: 4)

### Model Scenarios
- **baseline**: All features, no selection
- **baseline_pcc**: PCC for continuous features
- **baseline_chi2**: Chi-Square for discrete features  
- **baseline_pcc_chi2**: Combined PCC and Chi-Square selection

### Customization

Edit the following files to customize your setup:
- `requirements.txt`: Add/modify Python packages
- `docker-compose.yml`: Change ports or volume mounts
- `Dockerfile`: Modify the Python environment

### Troubleshooting

**Port already in use:**
```bash
docker-compose down
# Or change the port in docker-compose.yml
```

**Permission issues:**
The container runs as a non-root user for security. If you encounter permission issues, check file ownership.

**Memory issues:**
For large datasets, you may need to increase Docker's memory limit in Docker Desktop settings.