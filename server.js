const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;
const fs = require('fs');

const app = express();
const port = 3000;

// Middleware
app.use(bodyParser.json());
app.use(express.static('public'));

// Ensure directories exist
const dataDir = path.join(__dirname, 'data');
const analyticsDir = path.join(__dirname, 'analytic');
const assetsDir = path.join(__dirname, 'public', 'assets');

[dataDir, analyticsDir, assetsDir].forEach(dir => {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
});

// CSV Writer setup for cursor tracking
const cursorCsvWriter = createCsvWriter({
    path: path.join(dataDir, 'cursor_tracking.csv'),
    header: [
        { id: 'timestamp', title: 'TIMESTAMP' },
        { id: 'x', title: 'X_POSITION' },
        { id: 'y', title: 'Y_POSITION' },
        { id: 'isCutting', title: 'IS_CUTTING' },
        { id: 'score', title: 'SCORE' },
        { id: 'timeLeft', title: 'TIME_LEFT' },
        { id: 'level', title: 'LEVEL' },
        { id: 'fieldOfView', title: 'FIELD_OF_VIEW' },
        { id: 'distractionId', title: 'DISTRACTION_ID' },
        { id: 'distractionType', title: 'DISTRACTION_TYPE' },
        { id: 'distractionAction', title: 'DISTRACTION_ACTION' }
    ],
    append: true
});

// NEW: CSV Writer setup for vessel tracking
const vesselCsvWriter = createCsvWriter({
    path: path.join(dataDir, 'vessel_tracking.csv'),
    header: [
        { id: 'timestamp', title: 'TIMESTAMP' },
        { id: 'vesselId', title: 'VESSEL_ID' },
        { id: 'isCorrect', title: 'IS_CORRECT' },
        { id: 'startX', title: 'START_X' },
        { id: 'startY', title: 'START_Y' },
        { id: 'endX', title: 'END_X' },
        { id: 'endY', title: 'END_Y' },
        { id: 'cp1x', title: 'CONTROL_POINT1_X' },
        { id: 'cp1y', title: 'CONTROL_POINT1_Y' },
        { id: 'cp2x', title: 'CONTROL_POINT2_X' },
        { id: 'cp2y', title: 'CONTROL_POINT2_Y' },
        { id: 'pathPoints', title: 'PATH_POINTS' },
        { id: 'event', title: 'EVENT' },
        { id: 'isCut', title: 'IS_CUT' },
        { id: 'level', title: 'LEVEL' },
        { id: 'intertwined', title: 'IS_INTERTWINED' }
    ],
    append: true
});

// Route to serve the game
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// API endpoint to save cursor data
app.post('/api/cursor-log', async (req, res) => {
    try {
        const logData = req.body;
        await cursorCsvWriter.writeRecords(logData);
        res.json({ success: true });
    } catch (error) {
        console.error('Error writing to CSV:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// NEW: API endpoint to save vessel data
app.post('/api/vessel-log', async (req, res) => {
    try {
        const vesselData = req.body;
        await vesselCsvWriter.writeRecords(vesselData);
        res.json({ success: true });
    } catch (error) {
        console.error('Error writing vessel data to CSV:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// API endpoint to get analytics data
app.get('/api/analytics', async (req, res) => {
    try {
        const cursorData = fs.readFileSync(path.join(dataDir, 'cursor_tracking.csv'), 'utf8');
        const vesselData = fs.existsSync(path.join(dataDir, 'vessel_tracking.csv'))
            ? fs.readFileSync(path.join(dataDir, 'vessel_tracking.csv'), 'utf8')
            : '';

        res.json({
            cursorData,
            vesselData,
            hasVesselData: vesselData !== ''
        });
    } catch (error) {
        console.error('Error reading analytics data:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// API endpoint to get level-specific data
app.get('/api/analytics/level/:level', async (req, res) => {
    try {
        const level = parseInt(req.params.level);

        // Read cursor data
        const cursorData = fs.readFileSync(path.join(dataDir, 'cursor_tracking.csv'), 'utf8');
        const cursorLines = cursorData.split('\n');
        const levelCursorData = cursorLines.filter(line => {
            const columns = line.split(',');
            return columns.length >= 7 && parseInt(columns[6]) === level;
        });

        // Read vessel data
        let levelVesselData = [];
        if (fs.existsSync(path.join(dataDir, 'vessel_tracking.csv'))) {
            const vesselData = fs.readFileSync(path.join(dataDir, 'vessel_tracking.csv'), 'utf8');
            const vesselLines = vesselData.split('\n');
            levelVesselData = vesselLines.filter(line => {
                const columns = line.split(',');
                return columns.length >= 14 && parseInt(columns[13]) === level;
            });
        }

        res.json({
            cursorData: levelCursorData.join('\n'),
            vesselData: levelVesselData.join('\n'),
            hasVesselData: levelVesselData.length > 0
        });
    } catch (error) {
        console.error('Error reading level data:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// API endpoint to get vessel-specific data
app.get('/api/analytics/vessels', async (req, res) => {
    try {
        if (!fs.existsSync(path.join(dataDir, 'vessel_tracking.csv'))) {
            return res.json({
                data: '',
                message: 'No vessel data available yet.'
            });
        }

        const vesselData = fs.readFileSync(path.join(dataDir, 'vessel_tracking.csv'), 'utf8');
        res.json({ data: vesselData });
    } catch (error) {
        console.error('Error reading vessel data:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// API endpoint to get distraction-specific data
app.get('/api/analytics/distractions', async (req, res) => {
    try {
        const csvData = fs.readFileSync(path.join(dataDir, 'cursor_tracking.csv'), 'utf8');
        const lines = csvData.split('\n');

        // Filter lines that contain distraction data (columns 8-10 have values)
        const distractionData = lines.filter(line => {
            const columns = line.split(',');
            return columns.length >= 11 && columns[8] && columns[9] && columns[10];
        });

        res.json({ data: distractionData.join('\n') });
    } catch (error) {
        console.error('Error reading distraction data:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// API endpoint to calculate distraction reaction times
app.get('/api/analytics/reaction-times', async (req, res) => {
    try {
        // Note: We're not doing calculations here, just providing the raw data
        // as per requirements to avoid client-side computation
        const csvData = fs.readFileSync(path.join(dataDir, 'cursor_tracking.csv'), 'utf8');

        // Simply return all data; calculations will be done in the analytic stage
        res.json({
            message: "Raw data provided for processing. No calculations performed server-side.",
            data: csvData
        });
    } catch (error) {
        console.error('Error reading reaction time data:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// Start server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
    console.log(`Data directory: ${dataDir}`);
    console.log(`Analytics directory: ${analyticsDir}`);
    console.log(`Assets directory: ${assetsDir}`);
});
