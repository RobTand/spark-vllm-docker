#!/bin/bash
# Monitor AutoRound convergence and build histogram
# Usage: ./monitor_convergence.sh [container_name]

CONTAINER=${1:-amazing_bhabha}

echo "Monitoring convergence for container: $CONTAINER"
echo "Press Ctrl+C to stop"
echo ""

PREV_COUNT=0

while docker ps -q -f name=$CONTAINER | grep -q .; do
    COUNT=$(docker logs $CONTAINER 2>&1 | grep "quantized [0-9]" | wc -l)

    if [ "$COUNT" -gt "$PREV_COUNT" ]; then
        echo "=== $COUNT/64 layers complete ==="
        docker logs $CONTAINER 2>&1 | grep "quantized [0-9]" | sed 's/\x1b\[[0-9;]*m//g' | \
            awk -F'[: ,]' '{
                for(i=1;i<=NF;i++) {
                    if($i=="iter" && $(i-1)=="loss") start_loss=$(i+1)
                    if($i=="iter" && $(i+1)~/^[0-9]+$/ && $(i-1)=="->") { conv_iter=$(i+1); final_loss=$(i+2) }
                }
            } END {}'

        # Summary stats
        docker logs $CONTAINER 2>&1 | grep "quantized [0-9]" | sed 's/\x1b\[[0-9;]*m//g' | \
            grep -oP 'iter \K[0-9]+(?=:)' | awk '
            BEGIN { sum=0; max=0; min=999 }
            { sum+=$1; if($1>max)max=$1; if($1<min)min=$1; count++ }
            END { printf "  Convergence iters: min=%d, max=%d, avg=%.0f\n", min, max, sum/count }'

        docker logs $CONTAINER 2>&1 | grep "quantized [0-9]" | sed 's/\x1b\[[0-9;]*m//g' | \
            grep -oP '-> iter [0-9]+: \K[0-9.]+' | awk '
            BEGIN { sum=0; max=0; min=999 }
            { sum+=$1; if($1+0>max+0)max=$1; if($1+0<min+0)min=$1; count++ }
            END { printf "  Final loss: min=%s, max=%s, avg=%.7f\n", min, max, sum/count }'

        echo ""
        PREV_COUNT=$COUNT
    fi

    sleep 30
done

echo "Container stopped. Final summary:"
docker logs $CONTAINER 2>&1 | grep "quantized [0-9]" | sed 's/\x1b\[[0-9;]*m//g'
