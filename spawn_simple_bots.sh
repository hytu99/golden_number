#!/bin/bash

if [ -z $ROOM ]; then
    ROOM=1343
fi

trap kill_jobs INT

function kill_jobs () {
    echo "Killing..."
    kill $(jobs -p)
}

echo ROOM: $ROOM

for action in {0..7}; do
    python simple_bot.py --room $ROOM --action $action &
done

wait