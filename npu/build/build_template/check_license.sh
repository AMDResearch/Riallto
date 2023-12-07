set -o pipefail
set -e

source /opt/mlir_settings.sh

# Check if license variable is set
if [ -z "$XILINXD_LICENSE_FILE" ]; then
    echo "License env variable XILINXD_LICENSE_FILE is not set."
    echo "Please see https://riallto.ai/install-riallto.html for license setup instructions."
    exit 1
else
    # If variable set, see if file exists
    if [ ! -e "$XILINXD_LICENSE_FILE" ]; then
        echo "The license file $XILINXD_LICENSE_FILE does not exist."
        echo "Please see https://riallto.ai/install-riallto.html for license setup instructions."
        exit 1
    fi
fi