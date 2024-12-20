#!/bin/bash

# Define the starting and ending ports
START_PORT=8001
END_PORT=8004

# Iterate through the ports and start the servers
for PORT in $(seq $START_PORT $END_PORT); do
  echo "Starting server on port $PORT..."
  node index.js $PORT &
done

# Wait for all background processes to complete 
wait

