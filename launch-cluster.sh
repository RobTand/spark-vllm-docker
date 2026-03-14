#!/bin/bash

# Default Configuration
IMAGE_NAME="vllm-node"
DEFAULT_CONTAINER_NAME="vllm_node"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
# Modify these if you want to pass additional docker args or set VLLM_SPARK_EXTRA_DOCKER_ARGS variable
DOCKER_ARGS="-e NCCL_IGNORE_CPU_AFFINITY=1 -v $HF_CACHE_DIR:/root/.cache/huggingface"

# Append additional arguments from environment variable
if [[ -n "$VLLM_SPARK_EXTRA_DOCKER_ARGS" ]]; then
    DOCKER_ARGS="$DOCKER_ARGS $VLLM_SPARK_EXTRA_DOCKER_ARGS"
fi

# ETH_IF and IB_IF will be auto-detected if not provided
ETH_IF=""
IB_IF=""
NCCL_DEBUG_VAL=""

# Initialize variables
NODES_ARG=""
CONTAINER_NAME="$DEFAULT_CONTAINER_NAME"
COMMAND_TO_RUN=""
DAEMON_MODE="false"
CHECK_CONFIG="false"
ACTION="start"
CLUSTER_WAS_RUNNING="false"
MOD_PATHS=()
MOD_TYPES=()
LAUNCH_SCRIPT_PATH=""
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

ACTIONS_ARG=""
SOLO_MODE="false"
MOUNT_CACHE_DIRS="true"
BUILD_JOBS=""
NON_PRIVILEGED_MODE="false"
SERVER_MODE="false"
RESTART_POLICY=""
MEM_LIMIT_GB="110"
MEM_SWAP_LIMIT_GB=""
PIDS_LIMIT="4096"
SHM_SIZE_GB="64"

# Function to print usage
usage() {
    echo "Usage: $0 [-n <node_ips>] [-t <image_name>] [--name <container_name>] [--eth-if <if_name>] [--ib-if <if_name>] [--nccl-debug <level>] [--check-config] [--solo] [-d] [action] [command]"
    echo "  -n, --nodes     Comma-separated list of node IPs (Optional, auto-detected if omitted)"
    echo "  -t              Docker image name (Optional, default: $IMAGE_NAME)"
    echo "  --name          Container name (Optional, default: $DEFAULT_CONTAINER_NAME)"
    echo "  --eth-if        Ethernet interface (Optional, auto-detected)"
    echo "  --ib-if         InfiniBand interface (Optional, auto-detected)"
    echo "  -e, --env       Environment variable to pass to container (e.g. -e VAR=val)"
    echo "  -j              Number of parallel jobs for build environment variables (optional)"
    echo "  --nccl-debug    NCCL debug level (Optional, one of: VERSION, WARN, INFO, TRACE). If no level is provided, defaults to INFO."
    echo "  --apply-mod     Path to directory or zip file containing run.sh to apply before launch (Can be specified multiple times)"
    echo "  --launch-script Path to bash script to execute in the container (from examples/ directory or absolute path). If launch script is specified, action should be omitted."
    echo "  --check-config  Check configuration and auto-detection without launching"
    echo "  --solo          Solo mode: skip autodetection, launch only on current node, do not launch Ray cluster"
    echo "  --no-cache-dirs Do not mount default cache directories (~/.cache/vllm, ~/.cache/flashinfer, ~/.triton)"
    echo "  -d              Daemon mode (only for 'start' action)"
    echo "  --non-privileged Run in non-privileged mode (removes --privileged and --ipc=host)"
    echo "  --mem-limit-gb  Memory limit in GB (default: 110, only with --non-privileged)"
    echo "  --mem-swap-limit-gb Memory+swap limit in GB (default: mem-limit + 10, only with --non-privileged)"
    echo "  --pids-limit    Process limit (default: 4096, only with --non-privileged)"
    echo "  --shm-size-gb   Shared memory size in GB (default: 64, only with --non-privileged)"
    echo "  --server        Server mode: run vLLM as the container's main process (PID 1) so Docker"
    echo "                  restart policies apply. Implies --restart unless-stopped by default."
    echo "                  Only compatible with --solo and --launch-script."
    echo "  --restart <policy>  Docker restart policy (e.g. unless-stopped, always, on-failure:5)."
    echo "                      Automatically set by --server if not specified."
    echo "  action          start | stop | status | exec (Default: start). Not compatible with --launch-script."
    echo "  command         Command to run (only for 'exec' action). Not compatible with --launch-script."
    echo ""
    echo "Launch Script Usage:"
    echo "  $0 --launch-script examples/my-script.sh   # Script copied to container and executed"
    echo "  $0 --launch-script /path/to/script.sh      # Uses absolute path to script"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--nodes) NODES_ARG="$2"; shift ;;
        -t) IMAGE_NAME="$2"; shift ;;
        --name) CONTAINER_NAME="$2"; shift ;;
        --eth-if) ETH_IF="$2"; shift ;;
        --ib-if) IB_IF="$2"; shift ;;
        -e|--env) DOCKER_ARGS="$DOCKER_ARGS -e $2"; shift ;;
        -j) BUILD_JOBS="$2"; shift ;;
        --apply-mod) MOD_PATHS+=("$2"); shift ;;
        --launch-script) LAUNCH_SCRIPT_PATH="$2"; shift ;;
        --nccl-debug)
            if [[ -n "$2" && "$2" =~ ^(VERSION|WARN|INFO|TRACE)$ ]]; then
                NCCL_DEBUG_VAL="$2"
                shift
            else
                NCCL_DEBUG_VAL="INFO"
            fi
            ;;
        --check-config) CHECK_CONFIG="true" ;;
        --solo) SOLO_MODE="true" ;;
        --no-cache-dirs) MOUNT_CACHE_DIRS="false" ;;
        --non-privileged) NON_PRIVILEGED_MODE="true" ;;
        --server) SERVER_MODE="true" ;;
        --restart) RESTART_POLICY="$2"; shift ;;
        --mem-limit-gb) MEM_LIMIT_GB="$2"; shift ;;
        --mem-swap-limit-gb) MEM_SWAP_LIMIT_GB="$2"; shift ;;
        --pids-limit) PIDS_LIMIT="$2"; shift ;;
        --shm-size-gb) SHM_SIZE_GB="$2"; shift ;;
        -d) DAEMON_MODE="true" ;;
        -h|--help) usage ;;
        start|stop|status) 
            if [[ -n "$LAUNCH_SCRIPT_PATH" ]]; then
                echo "Error: Action '$1' is not compatible with --launch-script. Please omit the action or not use --launch-script."
                exit 1
            fi
            ACTION="$1" 
            ;;
        exec)
            if [[ -n "$LAUNCH_SCRIPT_PATH" ]]; then
                echo "Error: Action 'exec' is not compatible with --launch-script. Please omit the action or not use --launch-script."
                exit 1
            fi
            ACTION="exec"
            shift
            COMMAND_TO_RUN=$(printf "%q " "$@")
            break
            ;;
        *) 
            echo "Error: Unknown argument or action: $1"
            usage
            ;;
    esac
    shift
done

# Validate non-privileged mode flags
if [[ "$NON_PRIVILEGED_MODE" == "true" ]]; then
    # Set default swap limit if not specified
    if [[ -z "$MEM_SWAP_LIMIT_GB" ]]; then
        MEM_SWAP_LIMIT_GB=$((MEM_LIMIT_GB + 10))
    fi
else
    # Check if non-privileged flags were used without --non-privileged
    for flag in "--mem-limit-gb" "--mem-swap-limit-gb" "--pids-limit" "--shm-size-gb"; do
        if [[ "$*" == *"$flag"* ]]; then
            echo "Error: $flag can only be used with --non-privileged"
            exit 1
        fi
    done
fi

# Validate server mode
if [[ "$SERVER_MODE" == "true" ]]; then
    if [[ -z "$LAUNCH_SCRIPT_PATH" ]]; then
        echo "Error: --server requires --launch-script"
        exit 1
    fi
    # Default restart policy if not specified
    if [[ -z "$RESTART_POLICY" ]]; then
        RESTART_POLICY="unless-stopped"
    fi
fi

# Append NCCL_DEBUG if set, with validation
if [[ -n "$NCCL_DEBUG_VAL" ]]; then
    case "$NCCL_DEBUG_VAL" in
        VERSION|WARN|INFO|TRACE)
            DOCKER_ARGS="$DOCKER_ARGS -e NCCL_DEBUG=$NCCL_DEBUG_VAL"
            ;;
        *)
            echo "Error: Invalid value for --nccl-debug: $NCCL_DEBUG_VAL"
            echo "Allowed values: VERSION, WARN, INFO, TRACE"
            exit 1
            ;;
    esac
fi

# Add build job parallelization environment variables if BUILD_JOBS is set
if [[ -n "$BUILD_JOBS" ]]; then
    DOCKER_ARGS="$DOCKER_ARGS -e MAX_JOBS=$BUILD_JOBS"
    DOCKER_ARGS="$DOCKER_ARGS -e CMAKE_BUILD_PARALLEL_LEVEL=$BUILD_JOBS"
    DOCKER_ARGS="$DOCKER_ARGS -e NINJAFLAGS=-j$BUILD_JOBS"
    DOCKER_ARGS="$DOCKER_ARGS -e MAKEFLAGS=-j$BUILD_JOBS"
fi

# Add cache dirs if requested
CACHE_DIRS_TO_CREATE=()
if [[ "$MOUNT_CACHE_DIRS" == "true" ]]; then
    # vLLM Cache
    DOCKER_ARGS="$DOCKER_ARGS -v $HOME/.cache/vllm:/root/.cache/vllm"
    CACHE_DIRS_TO_CREATE+=("$HOME/.cache/vllm")
    
    # FlashInfer Cache
    DOCKER_ARGS="$DOCKER_ARGS -v $HOME/.cache/flashinfer:/root/.cache/flashinfer"
    CACHE_DIRS_TO_CREATE+=("$HOME/.cache/flashinfer")

    # Triton Cache
    DOCKER_ARGS="$DOCKER_ARGS -v $HOME/.triton:/root/.triton"
    CACHE_DIRS_TO_CREATE+=("$HOME/.triton")
fi

# Resolve launch script path if specified
if [[ -n "$LAUNCH_SCRIPT_PATH" ]]; then
    # Check if it's an absolute path or relative path that exists
    if [[ -f "$LAUNCH_SCRIPT_PATH" ]]; then
        LAUNCH_SCRIPT_PATH=$(realpath "$LAUNCH_SCRIPT_PATH")
    # Check if it's just a filename, look in examples/ directory
    elif [[ -f "$SCRIPT_DIR/examples/$LAUNCH_SCRIPT_PATH" ]]; then
        LAUNCH_SCRIPT_PATH="$SCRIPT_DIR/examples/$LAUNCH_SCRIPT_PATH"
    # Check if it's a name without .sh extension
    elif [[ -f "$SCRIPT_DIR/examples/${LAUNCH_SCRIPT_PATH}.sh" ]]; then
        LAUNCH_SCRIPT_PATH="$SCRIPT_DIR/examples/${LAUNCH_SCRIPT_PATH}.sh"
    else
        echo "Error: Launch script '$LAUNCH_SCRIPT_PATH' not found."
        echo "Searched in:"
        echo "  - $LAUNCH_SCRIPT_PATH"
        echo "  - $SCRIPT_DIR/examples/$LAUNCH_SCRIPT_PATH"
        echo "  - $SCRIPT_DIR/examples/${LAUNCH_SCRIPT_PATH}.sh"
        exit 1
    fi
    
    echo "Using launch script: $LAUNCH_SCRIPT_PATH"
    
    # Set command to run the copied script (use absolute path since docker exec may not be in /workspace)
    COMMAND_TO_RUN="/workspace/exec-script.sh"
    
    # If launch script is specified, default action to exec unless explicitly set to stop/status
    if [[ "$ACTION" == "start" ]]; then
        ACTION="exec"
    fi
fi

# Validate MOD_PATHS if set
for i in "${!MOD_PATHS[@]}"; do
    mod_path="${MOD_PATHS[$i]}"
    if [[ ! -e "$mod_path" ]]; then
        echo "Error: Mod path '$mod_path' does not exist."
        exit 1
    fi
    
    if [[ -d "$mod_path" ]]; then
        if [[ ! -f "$mod_path/run.sh" ]]; then
             echo "Error: Mod directory '$mod_path' must contain 'run.sh'."
             exit 1
        fi
        MOD_TYPES[$i]="dir"
    elif [[ -f "$mod_path" && "$mod_path" == *.zip ]]; then
        # Check zip content using unzip if available, else python
        if command -v unzip &> /dev/null; then
            if ! unzip -l "$mod_path" | grep -q "run.sh"; then
                 echo "Error: Mod zip file '$mod_path' must contain 'run.sh'."
                 exit 1
            fi
        else
             # Fallback to python for checking zip content
             if ! python3 -c "import zipfile, sys; sys.exit(0 if 'run.sh' in zipfile.ZipFile(sys.argv[1]).namelist() else 1)" "$mod_path"; then
                 echo "Error: Mod zip file '$mod_path' must contain 'run.sh'."
                 exit 1
             fi
        fi
        MOD_TYPES[$i]="zip"
    else
        echo "Error: --apply-mod '$mod_path' must be a directory or a .zip file."
        exit 1
    fi
    MOD_PATHS[$i]=$(realpath "$mod_path")
done

# --- Auto-Detection Logic ---
# Source autodiscover module
source "$(dirname "$0")/autodiscover.sh"

if [[ "$SOLO_MODE" == "true" ]]; then
    if [[ -n "$NODES_ARG" ]]; then
        echo "Error: --solo is incompatible with -n/--nodes."
        exit 1
    fi
    # Solo mode: skip node detection, just get local IP
    LOCAL_IP="127.0.0.1"
    NODES_ARG="$LOCAL_IP"
    PEER_NODES=()
    echo "Solo mode enabled. Skipping node detection."
else
    # Perform auto-detection
    detect_interfaces || exit 1
    detect_nodes || exit 1
fi

if [[ -z "$NODES_ARG" ]]; then
    echo "Error: Nodes argument (-n) is mandatory or could not be auto-detected."
    usage
fi

# Split nodes into array
IFS=',' read -r -a ALL_NODES <<< "$NODES_ARG"

if [[ "$SOLO_MODE" != "true" ]]; then
    # Detect Head IP (Local IP)
    detect_local_ip || exit 1
fi

HEAD_IP="$LOCAL_IP"

# Verify HEAD_IP is in ALL_NODES
FOUND_HEAD=false
for ip in "${ALL_NODES[@]}"; do
    ip=$(echo "$ip" | xargs)
    if [[ "$ip" == "$HEAD_IP" ]]; then
        FOUND_HEAD=true
        break
    fi
done

if [ "$FOUND_HEAD" = false ]; then
    echo "Error: Local IP ($HEAD_IP) is not in the list of nodes ($NODES_ARG)."
    exit 1
fi

# Implicit Solo Mode Detection
if [[ "$SOLO_MODE" == "false" && ${#PEER_NODES[@]} -eq 0 ]]; then
    echo "Only local node detected/configured. Activating solo mode (no Ray cluster)."
    SOLO_MODE="true"
fi

echo "Head Node: $HEAD_IP"
echo "Worker Nodes: ${PEER_NODES[*]}"
echo "Container Name: $CONTAINER_NAME"
echo "Image Name: $IMAGE_NAME"
echo "Action: $ACTION"

# Check SSH connectivity to worker nodes
if [[ "$ACTION" == "start" || "$ACTION" == "exec" || "$CHECK_CONFIG" == "true" ]]; then
    if [ ${#PEER_NODES[@]} -gt 0 ]; then
        echo "Checking SSH connectivity to worker nodes..."
        for worker in "${PEER_NODES[@]}"; do
            if ! ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$worker" true 2>/dev/null; then
                echo "Error: Passwordless SSH to $worker failed."
                echo "  Please ensure SSH keys are configured and the host is reachable."
                exit 1
            fi
            echo "  SSH to $worker: OK"
        done
    fi
fi

if [[ "$CHECK_CONFIG" == "true" ]]; then
    echo "Configuration Check Complete."
    echo "  Image Name: $IMAGE_NAME"
    echo "  ETH Interface: $ETH_IF"
    echo "  IB Interface: $IB_IF"
    echo "  Docker Args: $DOCKER_ARGS"
    if [[ "$MOUNT_CACHE_DIRS" == "true" ]]; then
         echo "  Mounting Cache Dirs: ${CACHE_DIRS_TO_CREATE[*]}"
    else
         echo "  Mounting Cache Dirs: (Disabled)"
    fi
    exit 0
fi

# Cleanup Function
cleanup() {
    # Remove traps to prevent nested cleanup
    trap - EXIT INT TERM HUP

    if [[ "$CLUSTER_WAS_RUNNING" == "true" ]]; then
        echo "Cluster was already running when script started. Skipping cleanup."
        return
    fi

    echo ""
    echo "Stopping cluster..."
    
    # Stop Head
    echo "Stopping head node ($HEAD_IP)..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    # Remove container if not using --rm (e.g. server mode with restart policy)
    if [[ -n "$RESTART_POLICY" ]]; then
        docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi

    # Stop Workers
    for worker in "${PEER_NODES[@]}"; do
        echo "Stopping worker node ($worker)..."
        ssh "$worker" "docker stop $CONTAINER_NAME" >/dev/null 2>&1 || true
        if [[ -n "$RESTART_POLICY" ]]; then
            ssh "$worker" "docker rm $CONTAINER_NAME" >/dev/null 2>&1 || true
        fi
    done
    
    echo "Cluster stopped."
}

# Handle 'stop' action
if [[ "$ACTION" == "stop" ]]; then
    cleanup
    exit 0
fi

# Handle 'status' action
if [[ "$ACTION" == "status" ]]; then
    echo "Checking status..."
    
    # Check Head
    if docker ps | grep -q "$CONTAINER_NAME"; then
        echo "[HEAD] $HEAD_IP: Container '$CONTAINER_NAME' is RUNNING."
        echo "--- Ray Status ---"
        docker exec "$CONTAINER_NAME" ray status || echo "Failed to get ray status."
        echo "------------------"
    else
        echo "[HEAD] $HEAD_IP: Container '$CONTAINER_NAME' is NOT running."
    fi
    
    # Check Workers
    for worker in "${PEER_NODES[@]}"; do
        if ssh "$worker" "docker ps | grep -q '$CONTAINER_NAME'"; then
             echo "[WORKER] $worker: Container '$CONTAINER_NAME' is RUNNING."
        else
             echo "[WORKER] $worker: Container '$CONTAINER_NAME' is NOT running."
        fi
    done
    exit 0
fi

# Trap signals
# Only trap if we are NOT in daemon mode (container should persist in daemon mode)
if [[ "$DAEMON_MODE" == "false" ]]; then
    trap cleanup EXIT INT TERM HUP
fi

# Check if cluster is already running
check_cluster_running() {
    local running=false
    
    # Check Head
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Warning: Container '$CONTAINER_NAME' is already running on head node ($HEAD_IP)."
        running=true
    fi
    
    # Check Workers
    for worker in "${PEER_NODES[@]}"; do
        if ssh "$worker" "docker ps --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}$'"; then
             echo "Warning: Container '$CONTAINER_NAME' is already running on worker node ($worker)."
             running=true
        fi
    done
    
    if [[ "$running" == "true" ]]; then
        echo "Cluster containers are already running. Skipping launch."
        CLUSTER_WAS_RUNNING="true"
        return 0
    fi
}

# Apply Mod Function
apply_mod_to_container() {
    local node_ip="$1"
    local container="$2"
    local is_local="$3" # true/false
    local mod_path="$4"
    local mod_type="$5"

    local mod_name=$(basename "$mod_path")
    if [[ "$mod_type" == "zip" ]]; then
        mod_name="${mod_name%.*}"
    fi

    echo "Applying mod '$mod_name' to $node_ip..."

    # 1. Copy mod to node (if remote)
    local target_mod_path=""
    local remote_cleanup_path=""

    if [[ "$is_local" == "true" ]]; then
        target_mod_path="$mod_path"
    else
        # SCP to remote
        local remote_tmp="/tmp/vllm_mod_pkg_$(date +%s)_$RANDOM"
        echo "  Copying mod package to $node_ip:$remote_tmp..."
        
        # Create directory first to ensure consistent path structure
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$node_ip" "mkdir -p $remote_tmp"
        remote_cleanup_path="$remote_tmp"
        
        if [[ "$mod_type" == "zip" ]]; then
             if ! scp -o BatchMode=yes -o StrictHostKeyChecking=no "$mod_path" "$node_ip:$remote_tmp/"; then
                echo "Error: Failed to copy mod to $node_ip"
                exit 1
             fi
             target_mod_path="$remote_tmp/$(basename "$mod_path")"
        else
             # Directory
             # Copy contents using wildcard to avoid creating a subdirectory
             if ! scp -r -o BatchMode=yes -o StrictHostKeyChecking=no "$mod_path"/* "$node_ip:$remote_tmp/"; then
                echo "Error: Failed to copy mod to $node_ip"
                exit 1
             fi
             target_mod_path="$remote_tmp"
        fi
    fi

    # 2. Copy into container
    local container_dest="/workspace/mods/$mod_name"
    
    # Command prefix for remote vs local
    local cmd_prefix=""
    if [[ "$is_local" == "false" ]]; then
        cmd_prefix="ssh -o BatchMode=yes -o StrictHostKeyChecking=no $node_ip"
    fi

    # Create workspace in container
    $cmd_prefix docker exec "$container" mkdir -p "$container_dest"

    if [[ "$mod_type" == "zip" ]]; then
        local zip_name=$(basename "$mod_path")
        echo "  Copying zip to container..."
        $cmd_prefix docker cp "$target_mod_path" "$container:$container_dest/$zip_name"
        
        # Unzip in container using python
        echo "  Extracting zip..."
        local py_unzip="import zipfile, sys; zipfile.ZipFile(sys.argv[1], 'r').extractall(sys.argv[2])"
        if [[ "$is_local" == "true" ]]; then
            docker exec "$container" python3 -c "$py_unzip" "$container_dest/$zip_name" "$container_dest"
        else
            $cmd_prefix docker exec "$container" python3 -c "\"$py_unzip\"" "$container_dest/$zip_name" "$container_dest"
        fi
    else
        # Directory
        echo "  Copying directory content to container..."
        if [[ "$is_local" == "true" ]]; then
             docker cp "$mod_path/." "$container:$container_dest/"
        else
             # For remote, we copied contents to $target_mod_path.
             # We want to copy contents of $target_mod_path to $container_dest.
             $cmd_prefix docker cp "$target_mod_path/." "$container:$container_dest/"
        fi
    fi

    # 3. Run run.sh
    echo "  Running patch script on $node_ip..."

    local local_exec_cmd="export WORKSPACE_DIR=\$PWD && cd $container_dest && chmod +x run.sh && ./run.sh"
    local remote_exec_cmd="export WORKSPACE_DIR=\\\$PWD && cd $container_dest && chmod +x run.sh && ./run.sh"
    local ret_code=0

    if [[ "$is_local" == "true" ]]; then
        docker exec "$container" bash -c "$local_exec_cmd"
        ret_code=$?
    else
        $cmd_prefix docker exec "$container" bash -c "\"$remote_exec_cmd\""
        ret_code=$?
    fi

    if [[ $ret_code -ne 0 ]]; then
        echo "Error: Patch script failed on $node_ip"
        # We should probably stop the cluster here or at least fail hard
        exit 1
    fi

    # 4. Cleanup remote temp
    if [[ "$is_local" == "false" ]]; then
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$node_ip" "rm -rf $remote_cleanup_path"
    fi
}

# Copy Launch Script to Container Function
copy_launch_script_to_container() {
    local container="$1"
    local script_path="$2"

    echo "Copying launch script to head node..."

    local target_script_path="$script_path"

    # Copy script into container as /workspace/exec-script.sh
    echo "  Copying script into container..."
    docker cp "$target_script_path" "$container:/workspace/exec-script.sh"

    # Make executable
    docker exec "$container" chmod +x /workspace/exec-script.sh

    echo "  Launch script copied to head node"
}

# Start Cluster Function
start_cluster() {
    check_cluster_running

    if [[ "$CLUSTER_WAS_RUNNING" == "true" ]]; then
        return
    fi

    # Start Head Node
    echo "Starting Head Node on $HEAD_IP..."
    
    # Ensure cache dirs exist on head
    if [[ "$MOUNT_CACHE_DIRS" == "true" ]]; then
        for dir in "${CACHE_DIRS_TO_CREATE[@]}"; do
            mkdir -p "$dir"
        done
    fi

    local head_cmd_args=()
    if [[ "$SOLO_MODE" == "true" && "$SERVER_MODE" == "true" ]]; then
        # Server mode: container waits for mods + launch script, then exec's vLLM as PID 1
        head_cmd_args=(bash -c "echo 'Server mode: waiting for launch script...'; while [ ! -f /workspace/exec-script.sh ] || [ ! -f /workspace/server_ready ]; do sleep 1; done; echo 'Launching vLLM as PID 1...'; exec bash /workspace/exec-script.sh")
    elif [[ "$SOLO_MODE" == "true" ]]; then
        if [[ ${#MOD_PATHS[@]} -gt 0 ]]; then
             head_cmd_args=(bash -c "echo Waiting for mod application...; while [ ! -f /tmp/mod_done ]; do sleep 1; done; echo Mod applied, starting container...; exec sleep infinity")
        else
             head_cmd_args=(sleep infinity)
        fi
    else
        if [[ ${#MOD_PATHS[@]} -gt 0 ]]; then
            head_cmd_args=(bash -c "echo Waiting for mod application...; while [ ! -f /tmp/mod_done ]; do sleep 1; done; echo Mod applied, starting node...; exec ./run-cluster-node.sh --role head --host-ip $HEAD_IP --eth-if $ETH_IF --ib-if $IB_IF")
        else
            head_cmd_args=(./run-cluster-node.sh --role head --host-ip "$HEAD_IP" --eth-if "$ETH_IF" --ib-if "$IB_IF")
        fi
    fi

    # Build docker run arguments based on mode
    local docker_rm_restart="--rm"
    if [[ -n "$RESTART_POLICY" ]]; then
        docker_rm_restart="--restart $RESTART_POLICY"
    fi
    local docker_args_common="--gpus all -d $docker_rm_restart --network host --name $CONTAINER_NAME $DOCKER_ARGS $IMAGE_NAME"
    local docker_caps_args=""
    local docker_resource_args=""

    if [[ "$NON_PRIVILEGED_MODE" == "true" ]]; then
        echo "Running in non-privileged mode..."
        docker_caps_args="--cap-add=IPC_LOCK"
        docker_resource_args="--shm-size=${SHM_SIZE_GB}g --device=/dev/infiniband --memory ${MEM_LIMIT_GB}g --memory-swap ${MEM_SWAP_LIMIT_GB}g --pids-limit ${PIDS_LIMIT}"
    else
        docker_caps_args="--privileged"
        docker_resource_args="--ipc=host"
    fi

    docker run $docker_caps_args $docker_resource_args \
        $docker_args_common \
        "${head_cmd_args[@]}"

    # Start Worker Nodes
    for worker in "${PEER_NODES[@]}"; do
        echo "Starting Worker Node on $worker..."
        
        # Ensure cache dirs exist on worker
        if [[ "$MOUNT_CACHE_DIRS" == "true" ]]; then
             # Create string of dirs to create
             dirs_str="${CACHE_DIRS_TO_CREATE[*]}"
             ssh "$worker" "mkdir -p $dirs_str"
        fi

        local docker_run_cmd="docker run $docker_caps_args $docker_resource_args $docker_args_common"
        
        if [[ ${#MOD_PATHS[@]} -gt 0 ]]; then
            local inner_script="echo Waiting for mod application...; while [ ! -f /tmp/mod_done ]; do sleep 1; done; echo Mod applied, starting node...; exec ./run-cluster-node.sh --role node --host-ip $worker --eth-if $ETH_IF --ib-if $IB_IF --head-ip $HEAD_IP"
            ssh "$worker" "$docker_run_cmd bash -c \"$inner_script\""
        else
            ssh "$worker" "$docker_run_cmd ./run-cluster-node.sh --role node --host-ip $worker --eth-if $ETH_IF --ib-if $IB_IF --head-ip $HEAD_IP"
        fi
    done

    # Apply mods if requested
    if [[ ${#MOD_PATHS[@]} -gt 0 ]]; then
        echo "Applying modifications to cluster nodes..."
        
        # Apply to Head
        for i in "${!MOD_PATHS[@]}"; do
            apply_mod_to_container "$HEAD_IP" "$CONTAINER_NAME" "true" "${MOD_PATHS[$i]}" "${MOD_TYPES[$i]}"
        done
        # Signal completion on Head
        docker exec "$CONTAINER_NAME" touch /tmp/mod_done
        
        # Apply to Workers
        for worker in "${PEER_NODES[@]}"; do
            for i in "${!MOD_PATHS[@]}"; do
                apply_mod_to_container "$worker" "$CONTAINER_NAME" "false" "${MOD_PATHS[$i]}" "${MOD_TYPES[$i]}"
            done
            # Signal completion on Worker
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$worker" "docker exec $CONTAINER_NAME touch /tmp/mod_done"
        done
    fi

    # Copy launch script to head node only (workers don't need it - they just run Ray)
    if [[ -n "$LAUNCH_SCRIPT_PATH" ]]; then
        copy_launch_script_to_container "$CONTAINER_NAME" "$LAUNCH_SCRIPT_PATH"
    fi

    if [[ "$SOLO_MODE" == "false" ]]; then
        wait_for_cluster
    else
        echo "Solo mode active: Skipping Ray cluster readiness check."
        # Give container a moment to start up
        sleep 2
    fi
}

# Wait for Cluster Readiness
wait_for_cluster() {
    echo "Waiting for cluster to be ready..."
    local retries=30
    local count=0
    
    while [[ $count -lt $retries ]]; do
        # Check if ray is responsive
        if docker exec "$CONTAINER_NAME" ray status >/dev/null 2>&1; then
             echo "Cluster head is responsive."
             # Give workers a moment to connect
             sleep 5
             return 0
        fi
        
        sleep 2
        ((count++))
    done
    
    echo "Timeout waiting for cluster to start."
    exit 1
}

if [[ "$ACTION" == "exec" ]]; then
    start_cluster
    echo "Executing command on head node: $COMMAND_TO_RUN"

    if [[ "$SERVER_MODE" == "true" ]]; then
        # Server mode: signal the container to exec the launch script as PID 1
        docker exec "$CONTAINER_NAME" touch /workspace/server_ready
        echo "Server mode: vLLM will start as the container's main process (PID 1)."
        echo "Container '$CONTAINER_NAME' has restart policy: ${RESTART_POLICY}"
        echo "Use 'docker logs -f $CONTAINER_NAME' to follow output."
    elif [[ "$DAEMON_MODE" == "true" ]]; then
        # Daemon mode: run command detached inside the container and exit immediately
        # Extract env vars starting from VLLM_HOST_IP to avoid interactive check in .bashrc
        # Redirect output to PID 1 stdout/stderr so it shows up in docker logs
        docker exec -d "$CONTAINER_NAME" bash -c "eval \"\$(sed -n '/export VLLM_HOST_IP/,\$p' /root/.bashrc)\" && { $COMMAND_TO_RUN; } >> /proc/1/fd/1 2>> /proc/1/fd/2"
        echo "Command dispatched in background (Daemon mode). Container: $CONTAINER_NAME"
    else
        # Check if running in a TTY to avoid "input device is not a TTY" error
        if [ -t 0 ]; then
            DOCKER_EXEC_FLAGS="-it"
        else
            DOCKER_EXEC_FLAGS="-i"
        fi

        docker exec $DOCKER_EXEC_FLAGS "$CONTAINER_NAME" bash -i -c "$COMMAND_TO_RUN"
    fi
elif [[ "$ACTION" == "start" ]]; then
    start_cluster
    if [[ "$DAEMON_MODE" == "true" ]]; then
        echo "Cluster started in background (Daemon mode)."
    else
        echo "Cluster started. Tailing logs from head node..."
        echo "Press Ctrl+C to stop the cluster."
        docker logs -f "$CONTAINER_NAME" &
        wait $!
    fi
fi
