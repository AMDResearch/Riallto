set -o pipefail
set -e

source /opt/mlir_settings.sh

# Check if license variable is set
if [ -z "$XILINXD_LICENSE_FILE" ]; then
    echo "License env variable XILINXD_LICENSE_FILE is not set."
    echo "Please see https://riallto.ai/install-riallto.html for license setup instructions."
    exit 1
else
    # Check if XILINXD_LICENSE_FILE is a file
    if [ -e "$XILINXD_LICENSE_FILE" ]; then
        echo "Found license file: $XILINXD_LICENSE_FILE"
    else
        # Assume XILINXD_LICENSE_FILE is a server, try to ping
        server_address=$(echo $XILINXD_LICENSE_FILE | cut -d@ -f2)
        if ping -c 1 $server_address &> /dev/null; then
            echo "Server $server_address is reachable."
        else
            echo "Could not find license file of reach server address $server_address."
            echo "Please see https://riallto.ai/install-riallto.html for license setup instructions."
            exit 1
        fi
    fi
fi
