#!/bin/bash

# Perfect Implementation Plan - Execution Script
# This script guides you through implementing first-place solutions

set -e  # Exit on error

COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_RED='\033[0;31m'
COLOR_BLUE='\033[0;34m'
COLOR_RESET='\033[0m'

MANTIS_DIR="/home/ocean/MANTIS"

echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
echo -e "${COLOR_BLUE}  MANTIS First Place Implementation${COLOR_RESET}"
echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
echo ""

# Function to print status
print_status() {
    echo -e "${COLOR_GREEN}✓${COLOR_RESET} $1"
}

print_warning() {
    echo -e "${COLOR_YELLOW}⚠${COLOR_RESET} $1"
}

print_error() {
    echo -e "${COLOR_RED}✗${COLOR_RESET} $1"
}

print_info() {
    echo -e "${COLOR_BLUE}ℹ${COLOR_RESET} $1"
}

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check Python
    if command -v python3 &> /dev/null; then
        print_status "Python3 found: $(python3 --version)"
    else
        print_error "Python3 not found"
        exit 1
    fi
    
    # Check required packages
    python3 -c "import tensorflow" 2>/dev/null && print_status "TensorFlow installed" || print_warning "TensorFlow not installed (pip install tensorflow)"
    python3 -c "import xgboost" 2>/dev/null && print_status "XGBoost installed" || print_warning "XGBoost not installed (pip install xgboost)"
    python3 -c "import lightgbm" 2>/dev/null && print_status "LightGBM installed" || print_warning "LightGBM not installed (pip install lightgbm)"
    
    # Check for data
    if [ -d "$MANTIS_DIR/data" ]; then
        print_status "Data directory exists"
    else
        print_warning "Data directory not found, creating..."
        mkdir -p "$MANTIS_DIR/data"
    fi
    
    echo ""
}

# Phase 1: Quick Wins
phase1_quick_wins() {
    echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
    echo -e "${COLOR_BLUE}  PHASE 1: Quick Wins (Days 1-3)${COLOR_RESET}"
    echo -e "${COLOR_BLUE}  Goal: Salience 0.10-0.15${COLOR_RESET}"
    echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
    echo ""
    
    # Day 1: Verify bug fix
    echo "Day 1: Verify Bug Fix"
    print_info "Retraining model with fixed class balance..."
    
    cd "$MANTIS_DIR"
    
    # Check if model architecture has the fix
    if grep -q "np.percentile" scripts/training/model_architecture.py; then
        print_status "Bug fix confirmed in model_architecture.py"
    else
        print_error "Bug fix not found! Check model_architecture.py line 353"
        exit 1
    fi
    
    # Retrain
    print_info "Starting training... (this may take a while)"
    print_warning "TODO: Run training manually:"
    echo "  python scripts/training/train_model.py --ticker XAUUSD"
    echo ""
    
    # Day 2-3: Calibration & Cross-Asset
    echo "Days 2-3: Add Calibration & Cross-Asset Features"
    
    # Check if calibration.py exists
    if [ -f "scripts/training/calibration.py" ]; then
        print_status "Calibration module exists"
    else
        print_error "Calibration module not found!"
        echo "Create it from PERFECT_IMPLEMENTATION_PLAN.md"
        exit 1
    fi
    
    # Create cross-asset features if not exists
    if [ ! -f "scripts/feature_engineering/cross_asset_features.py" ]; then
        print_info "Creating cross_asset_features.py..."
        print_warning "TODO: Copy code from PERFECT_IMPLEMENTATION_PLAN.md -> cross_asset_features.py"
    else
        print_status "Cross-asset features module exists"
    fi
    
    print_info "Phase 1 setup complete!"
    echo ""
}

# Phase 2: Major Upgrades
phase2_major_upgrades() {
    echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
    echo -e "${COLOR_BLUE}  PHASE 2: Major Upgrades (Days 4-10)${COLOR_RESET}"
    echo -e "${COLOR_BLUE}  Goal: Salience 0.20-0.30${COLOR_RESET}"
    echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
    echo ""
    
    # Days 4-6: Ensemble
    echo "Days 4-6: Model Ensemble"
    
    if [ ! -f "scripts/training/ensemble_model.py" ]; then
        print_warning "TODO: Create ensemble_model.py from PERFECT_IMPLEMENTATION_PLAN.md"
    else
        print_status "Ensemble module exists"
    fi
    
    # Days 7-10: Orthogonality
    echo "Days 7-10: Orthogonality Optimization (CRITICAL!)"
    
    if [ ! -f "scripts/training/orthogonality.py" ]; then
        print_warning "TODO: Create orthogonality.py from PERFECT_IMPLEMENTATION_PLAN.md"
        print_info "This is the BIGGEST impact (+0.10-0.15 salience)!"
    else
        print_status "Orthogonality module exists"
    fi
    
    print_info "Phase 2 modules ready for implementation"
    echo ""
}

# Phase 3: Testing
phase3_testing() {
    echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
    echo -e "${COLOR_BLUE}  PHASE 3: Testing (Days 11-14)${COLOR_RESET}"
    echo -e "${COLOR_BLUE}  Goal: Verify salience > 0.30${COLOR_RESET}"
    echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
    echo ""
    
    if [ ! -f "scripts/testing/local_salience_test.py" ]; then
        print_warning "TODO: Create local_salience_test.py from PERFECT_IMPLEMENTATION_PLAN.md"
    else
        print_status "Testing module exists"
    fi
    
    print_info "Testing checklist:"
    echo "  [ ] Test on ETHUSDT"
    echo "  [ ] Test on BTCUSDT"
    echo "  [ ] Test on XAUUSD"
    echo "  [ ] All saliences > 0.30"
    echo ""
}

# Phase 4: Deployment
phase4_deployment() {
    echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
    echo -e "${COLOR_BLUE}  PHASE 4: Deployment (Days 15-21)${COLOR_RESET}"
    echo -e "${COLOR_BLUE}  Goal: Deploy and take first place${COLOR_RESET}"
    echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
    echo ""
    
    print_info "Pre-deployment checklist:"
    echo "  [ ] Local salience > 0.30"
    echo "  [ ] All tests passing"
    echo "  [ ] Ensemble working"
    echo "  [ ] Calibration applied"
    echo "  [ ] Orthogonality optimized"
    echo "  [ ] R2 bucket configured"
    echo "  [ ] Hotkey ready"
    echo ""
    
    print_warning "Deployment steps:"
    echo "  1. Test in shadow mode"
    echo "  2. Register: btcli subnet register --netuid 123"
    echo "  3. Start mining: python miner.py --netuid 123"
    echo "  4. Monitor salience continuously"
    echo ""
}

# Main menu
show_menu() {
    echo ""
    echo "What would you like to do?"
    echo ""
    echo "1) Check Prerequisites"
    echo "2) Phase 1: Quick Wins (Days 1-3)"
    echo "3) Phase 2: Major Upgrades (Days 4-10)"
    echo "4) Phase 3: Testing (Days 11-14)"
    echo "5) Phase 4: Deployment (Days 15-21)"
    echo "6) Install Missing Dependencies"
    echo "7) View Implementation Plan"
    echo "8) Exit"
    echo ""
    read -p "Enter choice [1-8]: " choice
    
    case $choice in
        1) check_prerequisites ;;
        2) phase1_quick_wins ;;
        3) phase2_major_upgrades ;;
        4) phase3_testing ;;
        5) phase4_deployment ;;
        6) install_dependencies ;;
        7) view_plan ;;
        8) exit 0 ;;
        *) echo "Invalid choice" ;;
    esac
    
    show_menu
}

# Install dependencies
install_dependencies() {
    echo "Installing missing dependencies..."
    
    pip install --upgrade pip
    pip install tensorflow>=2.10.0
    pip install xgboost>=1.7.0
    pip install lightgbm>=3.3.0
    pip install scikit-learn>=1.0.0
    pip install pandas numpy scipy
    pip install optuna  # For hyperparameter optimization
    
    print_status "Dependencies installed!"
    echo ""
}

# View plan
view_plan() {
    if [ -f "$MANTIS_DIR/PERFECT_IMPLEMENTATION_PLAN.md" ]; then
        less "$MANTIS_DIR/PERFECT_IMPLEMENTATION_PLAN.md"
    else
        print_error "Implementation plan not found!"
    fi
}

# Main
main() {
    cd "$MANTIS_DIR" || exit 1
    
    echo ""
    print_info "Welcome to MANTIS First Place Implementation!"
    print_info "This script will guide you through the plan."
    echo ""
    
    print_info "Timeline: 2-4 weeks intensive work"
    print_info "Expected outcome: Salience 0.30-0.40 (first place)"
    echo ""
    
    show_menu
}

# Run
main

