class VesselGame {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');

        // Game state
        this.score = 0;
        this.timeLeft = 60;
        this.gameOver = false;
        this.isDrawing = false;
        this.cutPath = [];
        this.vessels = [];
        this.gameStarted = false;
        this.cursorLog = [];
        this.vesselLog = [];
        this.currentLevel = 1;
        this.vesselGenerationTimer = null;
        this.levelTimer = null;

        // Distractor elements
        this.distractors = [];
        this.distractorConfig = {
            minInterval: 5000,  // Minimum time between distractors (ms)
            maxInterval: 15000, // Maximum time between distractors (ms)
            types: ['blood_leak', 'warning_alert', 'instrument_request'],
            enabled: false
        };
        this.distractorTimer = null;

        // Background distractions
        this.backgroundDistractions = {
            calls: { active: false, startTime: null, duration: 0 },
            heartRateAlerts: { active: false, startTime: null, duration: 0 },
            voiceOvers: { active: false, startTime: null, duration: 0 }
        };
        this.backgroundDistractionConfig = {
            minInterval: 10000,  // Minimum time between background distractions (ms)
            maxInterval: 30000,  // Maximum time between background distractions (ms)
            enabled: false
        };
        this.backgroundDistractionTimer = null;

        // Level configurations
        this.levelConfig = {
            1: { vessels: 5, intertwined: false, fieldOfView: false, generationInterval: 100, distractors: false, backgroundDistractions: false },
            2: { vessels: 5, intertwined: false, fieldOfView: false, generationInterval: 100, distractors: false, backgroundDistractions: false },
            3: { vessels: 5, intertwined: false, fieldOfView: true, generationInterval: 100, distractors: true, backgroundDistractions: false },
            4: { vessels: 5, intertwined: true, fieldOfView: false, generationInterval: 100, distractors: true, backgroundDistractions: false },
            5: { vessels: 5, intertwined: true, fieldOfView: true, generationInterval: 100, distractors: true, backgroundDistractions: true },
            6: { vessels: 5, intertwined: true, fieldOfView: false, generationInterval: 100, distractors: true, backgroundDistractions: true },
            7: { vessels: 5, intertwined: true, fieldOfView: true, generationInterval: 100, distractors: true, backgroundDistractions: true }
        };

        // Field of view settings
        this.fieldOfView = {
            enabled: false,
            radius: 100
        };

        // Colors
        this.colors = {
            background: '#FF0000',
            vessel: '#808080',
            white: '#FFFFFF',
            black: '#000000',
            cutting: '#FFFF00',
            distractor: {
                blood_leak: '#8B0000',
                warning_alert: '#FFA500',
                instrument_request: '#4682B4'
            }
        };

        // Bind events
        this.bindEvents();

        this.cursorPos = { x: 0, y: 0 };

        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));

        // Initialize Tone.js sounds
        this.initToneSounds();
    }

    initToneSounds() {
        // Ensure Tone.js is initialized
        Tone.start();

        // Create synths for different sounds
        this.synths = {
            // Main game synth for cutting and feedback
            main: new Tone.PolySynth(Tone.Synth).toDestination(),

            // Synth for distractions and alerts
            alert: new Tone.Synth({
                oscillator: {
                    type: 'square'
                },
                envelope: {
                    attack: 0.01,
                    decay: 0.2,
                    sustain: 0.5,
                    release: 0.5
                }
            }).toDestination(),

            // Synth for background distractions
            background: new Tone.Synth({
                oscillator: {
                    type: 'sine'
                },
                envelope: {
                    attack: 0.1,
                    decay: 0.3,
                    sustain: 0.4,
                    release: 1
                }
            }).toDestination()
        };

        // Create effects
        this.effects = {
            // Reverb for ambient sounds
            reverb: new Tone.Reverb({
                decay: 2,
                wet: 0.3
            }).toDestination(),

            // Distortion for alerts
            distortion: new Tone.Distortion({
                distortion: 0.5,
                wet: 0.2
            }).toDestination(),

            // Filter for background sounds
            filter: new Tone.Filter({
                type: 'lowpass',
                frequency: 800
            }).toDestination()
        };

        // Connect effects
        this.synths.background.connect(this.effects.reverb);
        this.synths.alert.connect(this.effects.distortion);

        // Create sound sequences for background distractions
        this.sequences = {
            heartRateAlert: null,
            calls: null,
            voiceOvers: null
        };

        // Set up sequences
        this.setupSequences();
    }

    setupSequences() {
        // Heart rate alert sequence
        this.sequences.heartRateAlert = new Tone.Sequence((time, note) => {
            this.synths.alert.triggerAttackRelease(note, "16n", time);
        }, ["C4", "C4"], "4n");

        // Calls sequence
        this.sequences.calls = new Tone.Sequence((time, note) => {
            this.synths.background.triggerAttackRelease(note, "8n", time);
        }, ["E4", "A4", "E4", "A4"], "4n");

        // Voice overs - more complex melody
        this.sequences.voiceOvers = new Tone.Sequence((time, note) => {
            if (note !== null) {
                this.synths.main.triggerAttackRelease(note, "8n", time);
            }
        }, ["G3", "B3", "D4", null, "A3", "C4", null, "G3"], "8n");
    }

    playCorrectCutSound() {
        // Play a rising note sequence for correct cut
        this.synths.main.triggerAttackRelease("C4", "16n");
        setTimeout(() => this.synths.main.triggerAttackRelease("E4", "16n"), 100);
        setTimeout(() => this.synths.main.triggerAttackRelease("G4", "16n"), 200);
    }

    playWrongCutSound() {
        // Play a descending note sequence for wrong cut
        this.synths.main.triggerAttackRelease("E4", "16n");
        setTimeout(() => this.synths.main.triggerAttackRelease("D4", "16n"), 100);
        setTimeout(() => this.synths.main.triggerAttackRelease("C4", "16n"), 200);
    }

    playDistractorSound(type) {
        switch (type) {
            case 'blood_leak':
                this.synths.alert.triggerAttackRelease("C5", "8n");
                setTimeout(() => this.synths.alert.triggerAttackRelease("C5", "8n"), 300);
                break;
            case 'warning_alert':
                this.synths.alert.triggerAttackRelease("A4", "16n");
                setTimeout(() => this.synths.alert.triggerAttackRelease("A4", "16n"), 200);
                setTimeout(() => this.synths.alert.triggerAttackRelease("A4", "16n"), 400);
                break;
            case 'instrument_request':
                this.synths.alert.triggerAttackRelease("E4", "8n");
                setTimeout(() => this.synths.alert.triggerAttackRelease("G4", "8n"), 200);
                break;
        }
    }

    startBackgroundSound(type) {
        if (this.sequences[type]) {
            Tone.Transport.start();
            this.sequences[type].start(0);
        }
    }

    stopBackgroundSound(type) {
        if (this.sequences[type]) {
            this.sequences[type].stop();
        }
    }

    bindEvents() {
        this.canvas.addEventListener('mousedown', this.startCutting.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.endCutting.bind(this));
        this.canvas.addEventListener('mouseleave', this.endCutting.bind(this));
        this.canvas.addEventListener('click', this.handleClick.bind(this));
    }

    startGame() {
        this.resetGame();
        this.startLevel(1);
    }

    resetGame() {
        this.score = 0;
        this.gameStarted = true;
        this.gameOver = false;
        this.vessels = [];
        this.distractors = [];
        this.cursorLog = [];
        this.vesselLog = [];
        this.updateUI();

        // Reset all distraction states
        for (const type in this.backgroundDistractions) {
            this.backgroundDistractions[type].active = false;
            this.backgroundDistractions[type].startTime = null;
            this.stopBackgroundSound(type);
        }

        // Clear all timers
        if (this.distractorTimer) clearTimeout(this.distractorTimer);
        if (this.backgroundDistractionTimer) clearTimeout(this.backgroundDistractionTimer);
    }

    startLevel(level) {
        // Clear existing timers
        if (this.levelTimer) clearInterval(this.levelTimer);
        if (this.vesselGenerationTimer) clearInterval(this.vesselGenerationTimer);
        if (this.distractorTimer) clearTimeout(this.distractorTimer);
        if (this.backgroundDistractionTimer) clearTimeout(this.backgroundDistractionTimer);

        // Set up new level
        this.currentLevel = level;
        this.timeLeft = 60;
        this.vessels = [];
        this.distractors = [];
        const config = this.levelConfig[level];
        this.fieldOfView.enabled = config.fieldOfView;
        this.distractorConfig.enabled = config.distractors;
        this.backgroundDistractionConfig.enabled = config.backgroundDistractions;

        // Generate initial vessels
        this.generateInitialVessels(config);

        // Update UI
        this.updateUI();
        document.getElementById('levelComplete').classList.add('hidden');

        // Start level timer
        this.startLevelTimer();

        // Start distractor generation if enabled for this level
        if (this.distractorConfig.enabled) {
            this.scheduleNextDistractor();
        }

        // Start background distractions if enabled for this level
        if (this.backgroundDistractionConfig.enabled) {
            this.scheduleNextBackgroundDistraction();
        }
    }

    scheduleNextDistractor() {
        const delay = Math.floor(
            Math.random() *
            (this.distractorConfig.maxInterval - this.distractorConfig.minInterval) +
            this.distractorConfig.minInterval
        );

        this.distractorTimer = setTimeout(() => {
            if (this.gameStarted && !this.gameOver) {
                this.generateDistractor();
                this.scheduleNextDistractor();
            }
        }, delay);
    }

    scheduleNextBackgroundDistraction() {
        const delay = Math.floor(
            Math.random() *
            (this.backgroundDistractionConfig.maxInterval - this.backgroundDistractionConfig.minInterval) +
            this.backgroundDistractionConfig.minInterval
        );

        this.backgroundDistractionTimer = setTimeout(() => {
            if (this.gameStarted && !this.gameOver) {
                this.triggerRandomBackgroundDistraction();
                this.scheduleNextBackgroundDistraction();
            }
        }, delay);
    }

    generateDistractor() {
        if (!this.distractorConfig.enabled || this.distractors.length >= 3) return;

        // Choose a random distractor type
        const type = this.distractorConfig.types[
            Math.floor(Math.random() * this.distractorConfig.types.length)
        ];

        // Generate position outside field of view
        let x, y;
        if (this.fieldOfView.enabled) {
            // Calculate distance from cursor position that's outside FOV
            const distance = this.fieldOfView.radius + Math.random() * 200;
            const angle = Math.random() * Math.PI * 2;

            x = this.cursorPos.x + Math.cos(angle) * distance;
            y = this.cursorPos.y + Math.sin(angle) * distance;

            // Ensure coordinates are within canvas bounds
            x = Math.max(20, Math.min(this.canvas.width - 20, x));
            y = Math.max(20, Math.min(this.canvas.height - 20, y));
        } else {
            // If no FOV, place randomly on canvas, but away from cursor
            const minDistance = 200;
            do {
                x = Math.random() * (this.canvas.width - 40) + 20;
                y = Math.random() * (this.canvas.height - 40) + 20;
                const dx = x - this.cursorPos.x;
                const dy = y - this.cursorPos.y;
                var distance = Math.sqrt(dx * dx + dy * dy);
            } while (distance < minDistance);
        }

        // Create distractor object
        const distractor = {
            id: Date.now().toString(),
            type: type,
            x: x,
            y: y,
            radius: 15,
            appearTime: new Date().toISOString(), // Store creation time
            clicked: false
        };

        this.distractors.push(distractor);
        this.drawAll();

        // Play sound for distractor appearance
        this.playDistractorSound(type);

        // Log distractor appearance
        this.logDistractionEvent(distractor.id, distractor.type, 'appear', distractor.appearTime);
    }

    handleDistractorClick(distractor) {
        if (!distractor || distractor.clicked) return false;

        const clickTime = new Date().toISOString();
        distractor.clicked = true;
        distractor.clickTime = clickTime;

        // Log distractor click event
        this.logDistractionEvent(distractor.id, distractor.type, 'click', clickTime);

        // Show feedback
        this.showFeedback(`Distractor handled!`, distractor.x, distractor.y, '#00FF00');

        // Play acknowledgment sound
        this.synths.main.triggerAttackRelease("G4", "8n");

        return true;
    }

    triggerRandomBackgroundDistraction() {
        if (!this.backgroundDistractionConfig.enabled) return;

        // Choose a random distraction type
        const types = Object.keys(this.backgroundDistractions);
        const availableTypes = types.filter(type => !this.backgroundDistractions[type].active);

        if (availableTypes.length === 0) return;

        const type = availableTypes[Math.floor(Math.random() * availableTypes.length)];
        this.triggerBackgroundDistraction(type);
    }

    triggerBackgroundDistraction(type) {
        if (!this.backgroundDistractions[type] || this.backgroundDistractions[type].active) return;

        const startTime = new Date().toISOString();

        // Set distraction as active
        this.backgroundDistractions[type].active = true;
        this.backgroundDistractions[type].startTime = startTime;

        // Log start of background distraction
        this.logDistractionEvent('background', type, 'start', startTime);

        // Start background sound
        this.startBackgroundSound(type);

        // Show visual indicator
        document.getElementById('distraction-indicator').textContent = `${type.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ').trim()}`;
        document.getElementById('distraction-indicator').classList.remove('hidden');

        // Set duration based on type
        let duration = 0;
        switch (type) {
            case 'calls':
                duration = 10000; // 10 seconds
                break;
            case 'heartRateAlerts':
                duration = 8000; // 8 seconds
                break;
            case 'voiceOvers':
                duration = 6000; // 6 seconds
                break;
            default:
                duration = 5000; // 5 seconds
        }

        // Stop the distraction after duration
        setTimeout(() => {
            this.stopBackgroundDistraction(type);
        }, duration);
    }

    stopBackgroundDistraction(type) {
        if (!this.backgroundDistractions[type] || !this.backgroundDistractions[type].active) return;

        const endTime = new Date().toISOString();

        // Log end of background distraction
        this.logDistractionEvent('background', type, 'end', endTime);

        // Stop background sound
        this.stopBackgroundSound(type);

        // Calculate duration
        const startTime = new Date(this.backgroundDistractions[type].startTime);
        const duration = new Date(endTime) - startTime;
        this.backgroundDistractions[type].duration = duration;

        // Reset state
        this.backgroundDistractions[type].active = false;
        this.backgroundDistractions[type].startTime = null;

        // Hide visual indicator if no other distractions are active
        if (Object.keys(this.backgroundDistractions).every(t => !this.backgroundDistractions[t].active)) {
            document.getElementById('distraction-indicator').classList.add('hidden');
        }
    }

    logDistractionEvent(id, type, action, timestamp) {
        const logEntry = {
            timestamp,
            x: this.cursorPos.x,
            y: this.cursorPos.y,
            isCutting: this.isDrawing,
            score: this.score,
            timeLeft: this.timeLeft,
            level: this.currentLevel,
            fieldOfView: this.fieldOfView.enabled,
            distractionId: id,
            distractionType: type,
            distractionAction: action
        };

        this.cursorLog.push(logEntry);

        if (this.cursorLog.length >= 10) {
            this.saveCursorLog();
        }
    }

    startLevelTimer() {
        this.levelTimer = setInterval(() => {
            if (this.gameStarted) {
                this.timeLeft--;
                this.updateUI();

                if (this.timeLeft <= 0) {
                    if (this.currentLevel < 7) {
                        this.completeLevel();
                    } else {
                        this.endGame();
                    }
                }
            }
        }, 1000);
    }

    completeLevel() {
        clearInterval(this.levelTimer);
        clearInterval(this.vesselGenerationTimer);
        if (this.distractorTimer) clearTimeout(this.distractorTimer);
        if (this.backgroundDistractionTimer) clearTimeout(this.backgroundDistractionTimer);

        // Stop any active background distractions
        for (const type in this.backgroundDistractions) {
            if (this.backgroundDistractions[type].active) {
                this.stopBackgroundDistraction(type);
            }
        }

        // Play level complete sound
        this.synths.main.triggerAttackRelease("C4", "8n");
        setTimeout(() => this.synths.main.triggerAttackRelease("E4", "8n"), 200);
        setTimeout(() => this.synths.main.triggerAttackRelease("G4", "8n"), 400);
        setTimeout(() => this.synths.main.triggerAttackRelease("C5", "4n"), 600);

        document.getElementById('levelScore').textContent = this.score;
        document.getElementById('levelComplete').classList.remove('hidden');
    }

    nextLevel() {
        if (this.currentLevel < 7) {
            this.currentLevel++;
            this.startLevel(this.currentLevel);
        } else {
            this.endGame();
        }
    }

    generateInitialVessels(config) {
        // Clear existing vessels
        this.vessels = [];

        // Generate one correct vessel first
        this.generateNewVessel(true, config.intertwined);

        // Generate remaining vessels (4 more to make total of 5)
        for (let i = 1; i < config.vessels; i++) {
            this.generateNewVessel(false, config.intertwined);
        }
    }

    hasCorrectVessel() {
        return this.vessels.some(v => v.isCorrect && !v.isCut);
    }

    generateNewVessel(forceCorrect = false, intertwined = false) {
        // Clean up cut vessels if too many
        if (this.vessels.length > 5) {
            this.vessels = this.vessels.filter(v => !v.isCut);
        }

        const startX = Math.random() * (this.canvas.width - 200) + 100;
        const startY = Math.random() * (this.canvas.height - 200) + 100;
        const endX = Math.random() * (this.canvas.width - 200) + 100;
        const endY = intertwined ? startY : Math.random() * (this.canvas.height - 200) + 100;

        const cp1x = startX + (Math.random() - 0.5) * (intertwined ? 300 : 100);
        const cp1y = startY + (Math.random() - 0.5) * (intertwined ? 300 : 100);
        const cp2x = endX + (Math.random() - 0.5) * (intertwined ? 300 : 100);
        const cp2y = endY + (Math.random() - 0.5) * (intertwined ? 300 : 100);

        const points = [];
        for (let t = 0; t <= 1; t += 0.05) {
            const x = this.calculateBezierPoint(t, startX, cp1x, cp2x, endX);
            const y = this.calculateBezierPoint(t, startY, cp1y, cp2y, endY);
            points.push({ x, y });
        }

        const vesselId = Date.now().toString() + Math.floor(Math.random() * 1000);
        const isCorrect = forceCorrect ? true : Math.random() < 0.5;

        const vessel = {
            id: vesselId,
            start: { x: startX, y: startY },
            end: { x: endX, y: endY },
            cp1: { x: cp1x, y: cp1y },
            cp2: { x: cp2x, y: cp2y },
            points: points,
            isCorrect: isCorrect,
            isCut: false,
            creationTime: new Date().toISOString()
        };

        this.vessels.push(vessel);
        this.drawAll();

        // Log vessel creation to vessel log
        this.logVesselEvent(vessel, 'created');
    }

    logVesselEvent(vessel, event) {
        const timestamp = new Date().toISOString();

        // Serialize the path points to a JSON string for storage
        const pathPointsString = JSON.stringify(vessel.points.map(pt => ({ x: pt.x, y: pt.y })));

        const vesselData = {
            timestamp: timestamp,
            vesselId: vessel.id,
            isCorrect: vessel.isCorrect,
            startX: vessel.start.x,
            startY: vessel.start.y,
            endX: vessel.end.x,
            endY: vessel.end.y,
            cp1x: vessel.cp1.x,
            cp1y: vessel.cp1.y,
            cp2x: vessel.cp2.x,
            cp2y: vessel.cp2.y,
            pathPoints: pathPointsString,
            event: event,
            isCut: vessel.isCut,
            level: this.currentLevel,
            intertwined: this.levelConfig[this.currentLevel].intertwined
        };

        this.vesselLog.push(vesselData);

        // Save vessel log if it gets too large
        if (this.vesselLog.length >= 10) {
            this.saveVesselLog();
        }
    }

    async saveVesselLog() {
        if (this.vesselLog.length === 0) return;

        try {
            const response = await fetch('/api/vessel-log', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(this.vesselLog)
            });

            if (response.ok) {
                this.vesselLog = [];
            } else {
                console.error('Failed to save vessel log');
            }
        } catch (error) {
            console.error('Error saving vessel log:', error);
        }
    }

    calculateBezierPoint(t, p0, p1, p2, p3) {
        return Math.pow(1 - t, 3) * p0 +
            3 * Math.pow(1 - t, 2) * t * p1 +
            3 * (1 - t) * Math.pow(t, 2) * p2 +
            Math.pow(t, 3) * p3;
    }

    drawAll() {
        // Clear canvas
        this.ctx.fillStyle = this.colors.background;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        if (this.fieldOfView.enabled) {
            // Draw dark overlay
            this.ctx.fillStyle = 'rgba(0, 0, 0, 1)';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

            // Create spotlight effect at cursor position
            this.ctx.save();
            this.ctx.beginPath();
            this.ctx.arc(this.cursorPos.x, this.cursorPos.y, this.fieldOfView.radius, 0, Math.PI * 2);
            this.ctx.clip();
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

            // Redraw background and vessels in spotlight
            this.ctx.fillStyle = this.colors.background;
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            this.vessels.forEach(vessel => {
                if (!vessel.isCut) {
                    this.drawVessel(vessel);
                }
            });

            this.ctx.restore();
        } else {
            // Draw all vessels normally
            this.vessels.forEach(vessel => {
                if (!vessel.isCut) {
                    this.drawVessel(vessel);
                }
            });
        }

        // Draw all distractors
        this.distractors.forEach(distractor => {
            if (!distractor.clicked) {
                this.drawDistractor(distractor);
            }
        });

        // Draw cutting line if active
        if (this.isDrawing && this.cutPath.length > 1) {
            this.ctx.beginPath();
            this.ctx.moveTo(this.cutPath[0].x, this.cutPath[0].y);
            for (let point of this.cutPath) {
                this.ctx.lineTo(point.x, point.y);
            }
            this.ctx.strokeStyle = this.colors.cutting;
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        }

        // Draw any background distraction visual indicators
        this.drawBackgroundDistractionIndicators();
    }

    drawVessel(vessel) {
        this.ctx.beginPath();
        this.ctx.moveTo(vessel.start.x, vessel.start.y);
        this.ctx.bezierCurveTo(
            vessel.cp1.x, vessel.cp1.y,
            vessel.cp2.x, vessel.cp2.y,
            vessel.end.x, vessel.end.y
        );
        this.ctx.strokeStyle = this.colors.vessel;
        this.ctx.lineWidth = 5;
        this.ctx.stroke();

        // Draw endpoints
        this.drawEndpoint(vessel.start.x, vessel.start.y, this.colors.white);
        this.drawEndpoint(vessel.end.x, vessel.end.y,
            vessel.isCorrect ? this.colors.white : this.colors.black);
    }

    drawDistractor(distractor) {
        this.ctx.beginPath();
        this.ctx.arc(distractor.x, distractor.y, distractor.radius, 0, Math.PI * 2);
        this.ctx.fillStyle = this.colors.distractor[distractor.type] || '#FF00FF';
        this.ctx.fill();

        // Add a pulsing effect for attention
        const pulseSize = 5 * Math.sin(Date.now() / 200) + distractor.radius;
        this.ctx.beginPath();
        this.ctx.arc(distractor.x, distractor.y, pulseSize, 0, Math.PI * 2);
        this.ctx.strokeStyle = this.colors.distractor[distractor.type] || '#FF00FF';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();

        // Draw icon or symbol based on type
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';

        let symbol = '!';
        switch (distractor.type) {
            case 'blood_leak':
                symbol = '⚠';
                break;
            case 'warning_alert':
                symbol = '!';
                break;
            case 'instrument_request':
                symbol = '?';
                break;
        }

        this.ctx.fillText(symbol, distractor.x, distractor.y);
    }

    drawBackgroundDistractionIndicators() {
        // Draw visual indicators for active background distractions
        let y = 40;
        let activeDistractions = false;

        for (const type in this.backgroundDistractions) {
            if (this.backgroundDistractions[type].active) {
                activeDistractions = true;

                // Draw indicator based on type
                this.ctx.fillStyle = '#FFFFFF';
                this.ctx.font = '16px Arial';
                this.ctx.textAlign = 'left';
                this.ctx.textBaseline = 'middle';

                let displayText = '';
                switch (type) {
                    case 'calls':
                        displayText = '📞 Incoming Call';
                        break;
                    case 'heartRateAlerts':
                        displayText = '❤️ Heart Rate Alert';
                        break;
                    case 'voiceOvers':
                        displayText = '🔊 Voice Message';
                        break;
                }

                // Draw text with background
                const textWidth = this.ctx.measureText(displayText).width;
                this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                this.ctx.fillRect(10, y - 10, textWidth + 20, 30);

                this.ctx.fillStyle = '#FFFFFF';
                this.ctx.fillText(displayText, 20, y);

                y += 40;
            }
        }
    }

    drawEndpoint(x, y, color) {
        this.ctx.beginPath();
        this.ctx.arc(x, y, 5, 0, Math.PI * 2);
        this.ctx.fillStyle = color;
        this.ctx.fill();
    }

    handleMouseMove(event) {
        if (!this.gameStarted || this.gameOver) return;

        const rect = this.canvas.getBoundingClientRect();
        this.cursorPos = {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };

        // Log cursor position
        this.logCursorPosition(this.cursorPos.x, this.cursorPos.y, this.isDrawing);

        if (this.isDrawing) {
            this.continueCutting(this.cursorPos.x, this.cursorPos.y);
        }

        // Always redraw to update field of view
        this.drawAll();
    }

    handleClick(event) {
        if (!this.gameStarted || this.gameOver) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // Check if click hit a distractor
        for (let i = 0; i < this.distractors.length; i++) {
            const distractor = this.distractors[i];
            if (!distractor.clicked) {
                const dx = x - distractor.x;
                const dy = y - distractor.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance <= distractor.radius) {
                    this.handleDistractorClick(distractor);
                    // Remove clicked distractors after a short delay
                    setTimeout(() => {
                        this.distractors = this.distractors.filter(d => d.id !== distractor.id);
                        this.drawAll();
                    }, 1000);
                    return; // Don't process further if distractor was clicked
                }
            }
        }
    }

    getMousePosition() {
        return this.cutPath.length > 0 ? this.cutPath[this.cutPath.length - 1] : null;
    }

    startCutting(event) {
        if (!this.gameStarted || this.gameOver) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        this.isDrawing = true;
        this.cutPath = [{ x, y }];

        // Play cutting start sound
        // this.synths.main.triggerAttackRelease("A3", "32n");
    }

    checkAllVesselsCut() {
        // Check if all correct vessels are cut
        const allCorrectCut = this.vessels.every(vessel => vessel.isCorrect ? vessel.isCut : true);
        if (allCorrectCut) {
            this.generateNewVessels();
        }
    }

    generateNewVessels() {
        // Clear only the vessels, keep time and level unchanged
        this.vessels = [];

        // Generate new vessels according to the current level configuration
        const config = this.levelConfig[this.currentLevel];
        this.generateInitialVessels(config);

        // Redraw canvas with new vessels
        this.drawAll();
    }

    continueCutting(x, y) {
        this.cutPath.push({ x, y });
        this.drawAll();
    }

    endCutting() {
        if (!this.isDrawing) return;
        this.isDrawing = false;

        let cutMade = false;
        for (let vessel of this.vessels) {
            if (!vessel.isCut && this.checkCutIntersection(vessel)) {
                vessel.isCut = true;
                cutMade = true;

                // Log vessel cut event
                this.logVesselEvent(vessel, 'cut');

                if (vessel.isCorrect) {
                    this.score += 10;
                    this.showFeedback("Correct! +10", vessel.start.x, vessel.start.y, '#00FF00');
                    // this.playCorrectCutSound();
                } else {
                    this.score -= 5;
                    this.showFeedback("Wrong! -5", vessel.start.x, vessel.start.y, '#FF0000');
                    this.playWrongCutSound();
                }
            }
        }

        if (cutMade) {
            this.updateUI();
            // Check if we need to generate new set of vessels
            this.checkAllVesselsCut();
        }

        this.cutPath = [];
        this.drawAll();
    }

    checkCutIntersection(vessel) {
        if (this.cutPath.length < 2) return false;

        for (let i = 1; i < this.cutPath.length; i++) {
            const cut1 = this.cutPath[i - 1];
            const cut2 = this.cutPath[i];

            for (let j = 1; j < vessel.points.length; j++) {
                const vessel1 = vessel.points[j - 1];
                const vessel2 = vessel.points[j];

                if (this.lineIntersects(
                    cut1.x, cut1.y, cut2.x, cut2.y,
                    vessel1.x, vessel1.y, vessel2.x, vessel2.y
                )) {
                    return true;
                }
            }
        }
        return false;
    }

    lineIntersects(x1, y1, x2, y2, x3, y3, x4, y4) {
        const denominator = ((x2 - x1) * (y4 - y3)) - ((y2 - y1) * (x4 - x3));
        if (denominator === 0) return false;

        const ua = (((x4 - x3) * (y1 - y3)) - ((y4 - y3) * (x1 - x3))) / denominator;
        const ub = (((x2 - x1) * (y1 - y3)) - ((y2 - y1) * (x1 - x3))) / denominator;

        return (ua >= 0 && ua <= 1) && (ub >= 0 && ub <= 1);
    }

    showFeedback(text, x, y, color) {
        this.ctx.font = '20px Arial';
        this.ctx.fillStyle = color;
        this.ctx.fillText(text, x, y - 20);
    }

    updateUI() {
        document.getElementById('score').textContent = this.score;
        document.getElementById('timer').textContent = this.timeLeft;
        document.getElementById('currentLevel').textContent = this.currentLevel;
    }

    async logCursorPosition(x, y, isCutting) {
        const timestamp = new Date().toISOString();
        const logEntry = {
            timestamp,
            x,
            y,
            isCutting,
            score: this.score,
            timeLeft: this.timeLeft,
            level: this.currentLevel,
            fieldOfView: this.fieldOfView.enabled
        };
        this.cursorLog.push(logEntry);

        if (this.cursorLog.length >= 10) {
            await this.saveCursorLog();
        }
    }

    async saveCursorLog() {
        try {
            const response = await fetch('/api/cursor-log', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(this.cursorLog)
            });

            if (response.ok) {
                this.cursorLog = [];
            } else {
                console.error('Failed to save cursor log');
            }
        } catch (error) {
            console.error('Error saving cursor log:', error);
        }
    }

    async endGame() {
        // Clear all timers
        clearInterval(this.levelTimer);
        clearInterval(this.vesselGenerationTimer);
        if (this.distractorTimer) clearTimeout(this.distractorTimer);
        if (this.backgroundDistractionTimer) clearTimeout(this.backgroundDistractionTimer);

        // Stop any active background distractions
        for (const type in this.backgroundDistractions) {
            if (this.backgroundDistractions[type].active) {
                this.stopBackgroundDistraction(type);
            }
        }

        // Play game over sound
        this.synths.main.triggerAttackRelease("C5", "8n");
        setTimeout(() => this.synths.main.triggerAttackRelease("G4", "8n"), 200);
        setTimeout(() => this.synths.main.triggerAttackRelease("E4", "8n"), 400);
        setTimeout(() => this.synths.main.triggerAttackRelease("C4", "4n"), 600);

        this.gameStarted = false;
        this.gameOver = true;

        // Save any remaining cursor logs
        if (this.cursorLog.length > 0) {
            await this.saveCursorLog();
        }

        // Save any remaining vessel logs
        if (this.vesselLog.length > 0) {
            await this.saveVesselLog();
        }

        // Update UI
        document.getElementById('finalScore').textContent = this.score;
        document.getElementById('finalLevel').textContent = this.currentLevel;
        document.getElementById('gameOver').classList.remove('hidden');
        document.getElementById('startScreen').style.display = 'block';
    }

    cleanup() {
        clearInterval(this.levelTimer);
        clearInterval(this.vesselGenerationTimer);
        if (this.distractorTimer) clearTimeout(this.distractorTimer);
        if (this.backgroundDistractionTimer) clearTimeout(this.backgroundDistractionTimer);

        // Stop all sequences
        for (const type in this.sequences) {
            if (this.sequences[type]) {
                this.sequences[type].stop();
            }
        }

        Tone.Transport.stop();

        this.vessels = [];
        this.cutPath = [];
        this.distractors = [];
        this.cursorLog = [];
        this.vesselLog = [];

        // Stop any active background distractions
        for (const type in this.backgroundDistractions) {
            if (this.backgroundDistractions[type].active) {
                this.stopBackgroundDistraction(type);
            }
        }
    }
}

// Global game instance
let game;

function startNewGame() {
    if (game) {
        game.cleanup();
    }
    game = new VesselGame();
    game.startGame();
}

function nextLevel() {
    if (game) {
        game.nextLevel();
    }
}

// Initialize game when window loads
window.onload = () => {
    game = new VesselGame();
};
