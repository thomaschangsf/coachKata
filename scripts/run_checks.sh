#!/usr/bin/env bash
# Check script for running all code quality checks
# Usage: ./scripts/check.sh

# Don't use set -e here - we want to run all checks and report all failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section header
print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# Function to print success message
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Function to print error message
print_error() {
    echo -e "\n${RED}❌ $1${NC}\n"
}

# Function to print command being run
print_command() {
    echo -e "${YELLOW}Running: $1${NC}"
}

# Track if any check failed
FAILED=0
FAILED_COMMANDS=()

# Function to run a check command
run_check() {
    local name="$1"
    local command="$2"
    
    print_header "Running: $name"
    print_command "$command"
    
    # Capture both stdout and stderr
    if eval "$command" 2>&1; then
        print_success "$name passed"
        return 0
    else
        local exit_code=$?
        FAILED=1
        FAILED_COMMANDS+=("$name")
        print_error "$name failed (exit code: $exit_code)"
        echo -e "${YELLOW}To reproduce this failure, run:${NC}"
        echo -e "${YELLOW}  $command${NC}\n"
        return $exit_code
    fi
}

# Main execution
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                        Running Code Quality Checks                              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════════╝${NC}"

# Run all checks
run_check "ruff" "uv run ruff check --fix ."
run_check "pyright" "uv run pyright ."
run_check "pytest" "uv run pytest"
run_check "deptry" "uv run deptry ."

# Summary
echo ""
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                         🎉 All Checks Passed! 🎉                                 ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════════════╝${NC}\n"
    exit 0
else
    echo -e "${RED}╔══════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                          ❌ Some Checks Failed ❌                                 ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════════════════════════╝${NC}\n"
    echo -e "${YELLOW}Failed checks:${NC}"
    for cmd in "${FAILED_COMMANDS[@]}"; do
        echo -e "  ${RED}• $cmd${NC}"
    done
    echo ""
    echo -e "${YELLOW}To reproduce failures, run the commands shown above.${NC}\n"
    exit 1
fi
