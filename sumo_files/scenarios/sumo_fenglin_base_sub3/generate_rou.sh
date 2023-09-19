#!/bin/bash
for index in {1..1400}
do
  jtrrouter --route-files=base_${index}.flows.xml \
            --turn-ratio-files=input_turns.turns.xml \
            --net-file=base_v2.net.xml \
            --output-file=base_${index}.rou.xml \
            --accept-all-destinations=true  \
            --no-internal-links=true  \
            -e 86400
done