def get_html_template():
    return """
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🦠 Модель епідемії | Клітинний автомат</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                color: #333;
            }
            
            .container {
                max-width: 1600px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
                animation: fadeIn 0.8s ease-out;
                display: grid;
                grid-template-columns: 350px 1fr;
                gap: 20px;
                min-height: 80vh;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .header {
                text-align: center;
                margin-bottom: 20px;
                padding: 15px;
                background: linear-gradient(135deg, #ff6b6b, #ee5a24);
                border-radius: 15px;
                color: white;
                box-shadow: 0 10px 30px rgba(238, 90, 36, 0.3);
                grid-column: 1 / -1;
            }
            
            .header h1 {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            }
            
            .header p {
                font-size: 1.1rem;
                opacity: 0.9;
            }
            
            .sidebar {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            
            .controls-section {
                background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(0, 0, 0, 0.1);
                height: fit-content;
            }
            
            .controls-title {
                font-size: 1.3rem;
                font-weight: 600;
                color: #495057;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .controls {
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            
            .control-group {
                background: white;
                padding: 15px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .control-group:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            }
            
            label {
                display: block;
                font-weight: 600;
                color: #495057;
                margin-bottom: 10px;
                font-size: 0.95rem;
            }
            
            select, input[type="range"] {
                width: 100%;
                margin-bottom: 10px;
            }
            
            select {
                padding: 12px 15px;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                font-size: 1rem;
                background: white;
                transition: border-color 0.3s ease;
            }
            
            select:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            input[type="range"] {
                height: 8px;
                border-radius: 5px;
                background: #e9ecef;
                outline: none;
                -webkit-appearance: none;
            }
            
            input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none;
                appearance: none;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea, #764ba2);
                cursor: pointer;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
                transition: transform 0.2s ease;
            }
            
            input[type="range"]::-webkit-slider-thumb:hover {
                transform: scale(1.2);
            }
            
            .value-display {
                display: inline-block;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 5px 12px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.9rem;
                min-width: 50px;
                text-align: center;
            }
            
            .buttons-section {
                background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(0, 0, 0, 0.1);
            }
            
            .buttons {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            button {
                padding: 12px 20px;
                border: none;
                border-radius: 25px;
                font-size: 0.95rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                position: relative;
                overflow: hidden;
                width: 100%;
            }
            
            button:before {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 0;
                height: 0;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.3);
                transition: width 0.6s, height 0.6s, top 0.6s, left 0.6s;
                transform: translate(-50%, -50%);
            }
            
            button:active:before {
                width: 300px;
                height: 300px;
                top: 50%;
                left: 50%;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white;
            }
            
            .btn-success {
                background: linear-gradient(135deg, #28a745, #1e7e34);
                color: white;
            }
            
            .btn-danger {
                background: linear-gradient(135deg, #dc3545, #c82333);
                color: white;
            }
            
            .btn-secondary {
                background: linear-gradient(135deg, #6c757d, #495057);
                color: white;
            }
            
            button:hover:not(:disabled) {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            }
            
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }
            
            .status {
                text-align: center;
                padding: 15px 20px;
                border-radius: 12px;
                font-weight: 600;
                font-size: 1.1rem;
                margin-bottom: 20px;
                transition: all 0.3s ease;
            }
            
            .status.running {
                background: linear-gradient(135deg, #d4edda, #c3e6cb);
                color: #155724;
                border: 2px solid #28a745;
                animation: pulse 2s infinite;
            }
            
            .status.stopped {
                background: linear-gradient(135deg, #f8d7da, #f1b0b7);
                color: #721c24;
                border: 2px solid #dc3545;
            }
            
            .status.ready {
                background: linear-gradient(135deg, #cce7ff, #b3daff);
                color: #004085;
                border: 2px solid #007bff;
            }
            
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); }
                100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
            }
            
            .main-content {
                display: flex;
                flex-direction: column;
                gap: 20px;
                align-items: center;
                justify-content: center;
                min-height: 600px;
            }
            
            .visualization {
                text-align: center;
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
                width: 100%;
                max-width: 800px;
            }
            
            #simulationImage {
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                transition: transform 0.3s ease;
            }
            
            #simulationImage:hover {
                transform: scale(1.02);
            }
            
            .stats {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                padding: 20px;
                background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
                width: 100%;
                max-width: 800px;
            }
            
            .stat-item {
                background: white;
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
                transition: transform 0.3s ease;
            }
            
            .stat-item:hover {
                transform: translateY(-5px);
            }
            
            .stat-value {
                font-size: 2.5rem;
                font-weight: 800;
                margin-bottom: 8px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            .stat-label {
                font-size: 1rem;
                font-weight: 600;
                color: #6c757d;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .susceptible { color: #28a745; }
            .infected { color: #dc3545; }
            .recovered { color: #17a2b8; }
            
            @media (max-width: 1200px) {
                .container {
                    grid-template-columns: 1fr;
                    gap: 15px;
                    padding: 15px;
                    margin: 10px;
                }
                
                .sidebar {
                    order: 2;
                }
                
                .main-content {
                    order: 1;
                }
                
                .header h1 {
                    font-size: 2rem;
                }
                
                .controls {
                    flex-direction: row;
                    flex-wrap: wrap;
                }
                
                .control-group {
                    flex: 1;
                    min-width: 200px;
                }
                
                .buttons {
                    flex-direction: row;
                    flex-wrap: wrap;
                    gap: 10px;
                }
                
                button {
                    flex: 1;
                    min-width: 120px;
                }
                
                .stats {
                    grid-template-columns: repeat(auto-fit,
                    minmax(150px, 1fr));
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🦠 Клітинний автомат</h1>
                <p>Модель поширення епідемії (SIR)</p>
            </div>
            
            <div class="sidebar">
                <div class="controls-section">
                    <div class="controls-title">
                        ⚙️ Параметри симуляції
                    </div>
                    <div class="controls">
                        <div class="control-group">
                            <label for="gridSize">📏 Розмір сітки:</label>
                            <select id="gridSize">
                                <option value="30">30×30 (Швидко)</option>
                                <option value="50" selected>50×50 (Оптимально)
                                </option>
                                <option value="100">100×100 (Детально)</option>
                            </select>
                        </div>
                        <div class="control-group">
                            <label for="pInfect">🦠 Ймовірність зараження:
                            </label>
                            <input type="range" id="pInfect" min="0.1" 
                            max="1.0"
                                   step="0.1" value="0.3">
                            <span class="value-display" id="pInfectValue">0.3
                            </span>
                        </div>
                        <div class="control-group">
                            <label for="tRecover">⏱️ Час одужання (кроків):
                            </label>
                            <input type="range" id="tRecover" min="5" max="30"
                                   step="1" value="10">
                            <span class="value-display" id="tRecoverValue">10
                            </span>
                        </div>
                    </div>
                </div>
                
                <div class="buttons-section">
                    <div class="buttons">
                        <button class="btn-primary"
                         onclick="resetSimulation()">
                            🔄 Скидання
                        </button>
                        <button class="btn-success" onclick="startSimulation()"
                                id="startBtn">
                            ▶️ Запуск
                        </button>
                        <button class="btn-danger" onclick="stopSimulation()"
                                id="stopBtn" disabled>
                            ⏹️ Зупинка
                        </button>
                        <button class="btn-secondary"
                         onclick="stepSimulation()">
                            ⏭️ Один крок
                        </button>
                    </div>
                </div>
                
                <div class="status ready" id="status">🟢 Готовий до запуску
                </div>
            </div>
            
            <div class="main-content">
                <div class="visualization">
                    <img id="simulationImage" src=""
                         alt="Візуалізація симуляції" style="display: none;">
                </div>
                
                <div class="stats" id="stats" style="display: none;">
                    <div class="stat-item">
                        <div class="stat-value susceptible"
                             id="susceptibleCount">0</div>
                        <div class="stat-label">Сприйнятливі</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value infected"
                             id="infectedCount">0</div>
                        <div class="stat-label">Інфіковані</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value recovered"
                             id="recoveredCount">0</div>
                        <div class="stat-label">Одужалі</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let isRunning = false;
            let stepCount = 0;
            let intervalId = null;

            document.getElementById('pInfect').oninput = function() {
                document.getElementById('pInfectValue')
                    .textContent = this.value;
            };
            document.getElementById('tRecover').oninput = function() {
                document.getElementById('tRecoverValue')
                    .textContent = this.value;
            };

                        async function makeRequest(url, data = null) {
                const options = {
                    method: (data !== null || url === '/step') 
                        ? 'POST' : 'GET',
                    headers: { 'Content-Type': 'application/json' }
                };
                if (data) {
                    options.body = JSON.stringify(data);
                }
                
                try {
                    const response = await fetch(url, options);
                    if (!response.ok) {
                        throw new Error(
                            `HTTP error! status: ${response.status}`);
                    }
                    return await response.json();
                } catch (error) {
                    console.error('Request failed:', error);
                    throw error;
                }
            }

            function updateDisplay(data) {
                const img = document.getElementById('simulationImage');
                img.src = 'data:image/png;base64,' + data.image;
                img.style.display = 'block';
                
                const stats = data.stats;
                document.getElementById('susceptibleCount')
                    .textContent = stats.susceptible.toLocaleString();
                document.getElementById('infectedCount')
                    .textContent = stats.infected.toLocaleString();
                document.getElementById('recoveredCount')
                    .textContent = stats.recovered.toLocaleString();
                document.getElementById('stats').style.display = 'grid';
                
                stepCount++;
                
                if (isRunning) {
                    updateStatus(
                        `🔄 Крок ${stepCount} | Інфікованих: ` +
                        `${stats.infected.toLocaleString()}`, 'running');
                    
                    if (stats.infected === 0) {
                        stopSimulation();
                        updateStatus('✅ Епідемія завершена', 'stopped');
                    }
                } else {
                    updateStatus(
                        `📊 Крок ${stepCount} | Інфікованих: ` +
                        `${stats.infected.toLocaleString()}`, 'ready');
                }
            }

            function updateStatus(message, type = 'ready') {
                const status = document.getElementById('status');
                status.textContent = message;
                status.className = `status ${type}`;
            }

                        async function resetSimulation() {
                stopSimulation();
                
                const params = {
                    size: parseInt(
                        document.getElementById('gridSize').value),
                    p_infect: parseFloat(
                        document.getElementById('pInfect').value),
                    t_recover: parseInt(
                        document.getElementById('tRecover').value)
                };
                
                try {
                    updateStatus('⏳ Скидання симуляції...', 'ready');
                    const data = await makeRequest('/reset', params);
                    updateDisplay(data);
                    stepCount = 0;
                    updateStatus('🔄 Симуляцію скинуто', 'ready');
                } catch (error) {
                    console.error('Error:', error);
                    updateStatus('❌ Помилка скидання', 'stopped');
                }
            }

            function startSimulation() {
                if (isRunning) return;
                
                isRunning = true;
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                updateStatus('🚀 Симуляцію запущено', 'running');
                
                intervalId = setInterval(async () => {
                    try {
                        const data = await makeRequest('/step');
                        updateDisplay(data);
                    } catch (error) {
                        console.error('Step error:', error);
                        stopSimulation();
                        updateStatus('❌ Помилка виконання кроку', 'stopped');
                    }
                }, 300);
            }

            function stopSimulation() {
                if (!isRunning && !intervalId) return;
                
                isRunning = false;
                
                if (intervalId) {
                    clearInterval(intervalId);
                    intervalId = null;
                }
                
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                
                const currentStats = {
                    infected: parseInt(
                        document.getElementById('infectedCount')
                            .textContent.replace(/,/g, '') || 0)
                };
                
                if (currentStats.infected === 0) {
                    updateStatus('✅ Епідемія завершена', 'stopped');
                } else {
                    updateStatus('⏸️ Симуляцію зупинено', 'stopped');
                }
            }

            async function stepSimulation() {
                if (isRunning) return;
                
                try {
                    updateStatus('⏳ Виконання кроку...', 'ready');
                    const data = await makeRequest('/step');
                    updateDisplay(data);
                } catch (error) {
                    console.error('Error:', error);
                    updateStatus('❌ Помилка виконання кроку', 'stopped');
                }
            }

            window.onload = function() {
                updateStatus('⏳ Ініціалізація...', 'ready');
                resetSimulation();
            };
        </script>
    </body>
    </html>
    """ 