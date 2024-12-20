const http = require('http');

// Get the port from the command line arguments
const PORT = process.argv[2] || 8080; // Default to 8080 if no port is provided

// Create the server
const server = http.createServer((req, res) => {
    console.log("received request");
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('Hello World\n');
});

// Start the server on the specified port
server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}/`);
});

