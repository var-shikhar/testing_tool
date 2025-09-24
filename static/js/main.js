document.addEventListener('DOMContentLoaded', () => {
    // --- GLOBAL SELECTORS ---
    const tabs = document.querySelectorAll('.tab-link');
    const tabContents = document.querySelectorAll('.tab-content');

    // --- GENERATION TAB SELECTORS ---
    const instructionsArea = document.getElementById('instructions-area');
    const saveInstructionsBtn = document.getElementById('save-instructions-btn');
    const requirementArea = document.getElementById('requirement-area');
    const generateBtn = document.getElementById('generate-btn');
    const tcLoader = document.getElementById('tc-loader');
    const generatedHeading = document.getElementById('generated-heading');
    const generatedSteps = document.getElementById('generated-steps');
    const generatedVerification = document.getElementById('generated-verification');
    const passStatusSelect = document.getElementById('pass-status-select');
    const saveTcBtn = document.getElementById('save-tc-btn');
    const testCaseListContainer = document.getElementById('test-case-list-container');
    const exportBtn = document.getElementById('export-btn');
    const deleteBtn = document.getElementById('delete-btn');

    // --- EXECUTION TAB SELECTORS ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results-section');
    const statusMessage = document.getElementById('status-message');
    const downloadLinkContainer = document.getElementById('download-link-container');
    const logContainer = document.getElementById('log-container');

    // --- CHAT TAB SELECTORS ---
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const recordButton = document.getElementById('recordButton');
    const recordingStatus = document.getElementById('recordingStatus');

    // --- MODAL SELECTORS ---
    const modalBackdrop = document.getElementById('modal-backdrop');
    const modalImage = document.getElementById('modal-image');
    const closeModal = document.getElementById('close-modal');

    // =====================================================================
    // TAB SWITCHING LOGIC
    // =====================================================================
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(item => item.classList.remove('active'));
            tab.classList.add('active');
            const target = document.getElementById(tab.dataset.tab);
            tabContents.forEach(content => content.classList.remove('active'));
            target.classList.add('active');
        });
    });

    // =====================================================================
    // GENERATION TAB LOGIC
    // =====================================================================
    const API_PREFIX = '/generate';

    // --- Instructions ---
    async function loadInstructions() {
        try {
            const response = await fetch(`${API_PREFIX}/instructions`);
            const data = await response.json();
            instructionsArea.value = data.text || '';
        } catch (error) {
            console.error('Failed to load instructions:', error);
        }
    }

    saveInstructionsBtn.addEventListener('click', async () => {
        try {
            await fetch(`${API_PREFIX}/instructions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: instructionsArea.value })
            });
            alert('Instructions saved!');
        } catch (error) {
            console.error('Failed to save instructions:', error);
            alert('Error saving instructions.');
        }
    });

    // --- Test Case Generation ---
    generateBtn.addEventListener('click', async () => {
        const requirement = requirementArea.value.trim();
        if (!requirement) {
            alert('Please enter a primary requirement.');
            return;
        }
        tcLoader.classList.remove('hidden');
        generateBtn.disabled = true;
        try {
            const response = await fetch(`${API_PREFIX}/generate_test_cases`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: requirement,
                    additional_instructions: instructionsArea.value
                })
            });
            if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
            const data = await response.json();
            generatedHeading.value = data.heading || '';
            generatedSteps.value = data.steps || '';
            generatedVerification.value = data.verification || '';
            saveTcBtn.disabled = false;
        } catch (error) {
            console.error('Failed to generate test case:', error);
            alert('Error generating test case. See console for details.');
        } finally {
            tcLoader.classList.add('hidden');
            generateBtn.disabled = false;
        }
    });

    // --- Saving & Listing Test Cases ---
    saveTcBtn.addEventListener('click', async () => {
        const testCase = {
            heading: generatedHeading.value,
            steps: generatedSteps.value,
            verification: generatedVerification.value,
            pass_status: passStatusSelect.value
        };
        try {
            await fetch(`${API_PREFIX}/test_cases`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(testCase)
            });
            // Clear fields and refresh list
            generatedHeading.value = '';
            generatedSteps.value = '';
            generatedVerification.value = '';
            saveTcBtn.disabled = true;
            loadAllTestCases();
        } catch (error) {
            console.error('Failed to save test case:', error);
        }
    });

    async function loadAllTestCases() {
        try {
            const response = await fetch(`${API_PREFIX}/test_cases`);
            const testCases = await response.json();
            renderTestCaseList(testCases.reverse());
        } catch (error) {
            console.error('Failed to load test cases:', error);
        }
    }

    function renderTestCaseList(testCases) {
        if (!testCases.length) {
            testCaseListContainer.innerHTML = '<p class="empty-list">No saved test cases yet.</p>';
            return;
        }
        let tableHTML = `
            <table>
                <thead>
                    <tr>
                        <th><input type="checkbox" id="select-all-cb"></th>
                        <th>Heading</th>
                        <th>Pass Status</th>
                        <th>Last Modified</th>
                    </tr>
                </thead>
                <tbody>
        `;
        testCases.forEach(tc => {
            tableHTML += `
                <tr data-id="${tc.id}">
                    <td><input type="checkbox" class="tc-checkbox" value="${tc.id}"></td>
                    <td>${tc.heading}</td>
                    <td>${tc.pass_status === 'M' ? 'Mandatory' : 'Optional'}</td>
                    <td>${new Date(tc.last_modified).toLocaleString()}</td>
                </tr>
            `;
        });
        tableHTML += '</tbody></table>';
        testCaseListContainer.innerHTML = tableHTML;

        // Add event listener for the new "select all" checkbox
        document.getElementById('select-all-cb').addEventListener('change', (e) => {
            document.querySelectorAll('.tc-checkbox').forEach(cb => {
                cb.checked = e.target.checked;
            });
        });
    }

    // --- Delete & Export ---
    deleteBtn.addEventListener('click', async () => {
        const selectedIds = Array.from(document.querySelectorAll('.tc-checkbox:checked')).map(cb => cb.value);
        if (selectedIds.length === 0) {
            alert('Please select test cases to delete.');
            return;
        }
        if (!confirm(`Are you sure you want to delete ${selectedIds.length} test case(s)?`)) return;

        for (const id of selectedIds) {
            await fetch(`${API_PREFIX}/test_cases/${id}`, { method: 'DELETE' });
        }
        loadAllTestCases();
    });

    exportBtn.addEventListener('click', async () => {
        const selectedIds = Array.from(document.querySelectorAll('.tc-checkbox:checked')).map(cb => cb.value);
        if (selectedIds.length === 0) {
            alert('Please select test cases to export.');
            return;
        }
        try {
            const response = await fetch(`${API_PREFIX}/export_xlsx`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ test_case_ids: selectedIds })
            });
            if (!response.ok) throw new Error('Export failed');
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'test_cases.xlsx';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Export failed:', error);
        }
    });

    // =====================================================================
    // EXECUTION TAB LOGIC
    // =====================================================================
    let executionEventSource; // Keep a separate EventSource for the execution tab

    // --- Drag & Drop ---
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleExecFileSubmit();
        }
    });
    fileInput.addEventListener('change', () => { if (fileInput.files.length) handleExecFileSubmit(); });

    // --- File Submission & WebSocket ---
    async function handleExecFileSubmit() {
        uploadForm.classList.add('hidden');
        loader.classList.remove('hidden');
        resultsSection.classList.remove('hidden');
        logContainer.innerHTML = '';
        downloadLinkContainer.innerHTML = '';
        statusMessage.textContent = 'Uploading file and preparing test run...';

        const formData = new FormData(uploadForm);
        try {
            const response = await fetch('/execution/upload-and-run-test/', { method: 'POST', body: formData });
            if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
            const data = await response.json();
            statusMessage.textContent = 'Connection established. Test is now running...';
            connectExecutionWebSocket(data.run_id);
        } catch (error) {
            statusMessage.textContent = `Error: ${error.message}`;
            loader.classList.add('hidden');
            uploadForm.classList.remove('hidden');
        }
    }

    function connectExecutionWebSocket(runId) {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // **** THE FIX IS HERE: Changed /executions/ to /execution/ ****
        const ws = new WebSocket(`${wsProtocol}//127.0.0.1:8000/execution/ws/${runId}`);

        ws.onopen = () => console.log('Execution WebSocket connection established.');

        ws.onmessage = (event) => handleExecutionWebSocketMessage(JSON.parse(event.data));

        ws.onclose = () => {
            statusMessage.textContent = 'Test run finished or connection lost.';
            loader.classList.add('hidden');
            uploadForm.classList.remove('hidden');
        };

        ws.onerror = (error) => {
            console.error('Execution WebSocket error:', error);
            statusMessage.textContent = 'Execution WebSocket connection error.';
            loader.classList.add('hidden');
        };
    }

    function handleExecutionWebSocketMessage(data) {
        const isScrolledToBottom = logContainer.scrollHeight - logContainer.clientHeight <= logContainer.scrollTop + 5;
        if (data.type === 'log') {
            const logElement = document.createElement('div');
            logElement.className = `log-item level-${data.level}`;
            if (data.is_header) logElement.classList.add('level-is_header');
            logElement.textContent = data.message;
            if (data.level === 'result' && data.status) {
                const statusP = document.createElement('p');
                statusP.innerHTML = `<strong>Final Status:</strong> ${data.status}`;
                if (data.status.toLowerCase().startsWith('failed')) statusP.classList.add('status-failed');
                logElement.appendChild(statusP);
            }
            logContainer.appendChild(logElement);
        } else if (data.type === 'screenshot') {
            const screenshotDiv = document.createElement('div');
            screenshotDiv.className = 'screenshot-container';
            screenshotDiv.innerHTML = `<p><i class="fa-solid fa-camera"></i> Screenshot for TC ${data.s_no} / Step ${data.step}:</p>`;
            const link = document.createElement('a');
            link.href = data.path;
            const img = document.createElement('img');
            img.src = data.path;
            link.appendChild(img);
            screenshotDiv.appendChild(link);
            logContainer.appendChild(screenshotDiv);
            link.addEventListener('click', (e) => {
                e.preventDefault();
                modalImage.src = data.path;
                modalBackdrop.style.display = 'flex';
            });
        } else if (data.type === 'result_file') {
            loader.classList.add('hidden');
            const downloadLink = document.createElement('a');
            downloadLink.href = data.path;
            downloadLink.setAttribute('download', data.filename);
            downloadLink.innerHTML = `<i class="fa-solid fa-file-excel"></i> Download Results (${data.filename})`;
            downloadLinkContainer.innerHTML = '';
            downloadLinkContainer.appendChild(downloadLink);
            statusMessage.textContent = 'Test run complete! Results are ready.';
        }
        if (isScrolledToBottom) logContainer.scrollTop = logContainer.scrollHeight;
    }

    // =====================================================================
    // CHAT TAB LOGIC
    // =====================================================================
    let mediaRecorder;
    let audioChunks = [];
    let chatEventSource;
    let chatClientId = `webClient-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;

    function connectChatEventSource() {
        if (chatEventSource) {
            chatEventSource.close();
        }
        chatEventSource = new EventSource(`/chat/chat_stream/${chatClientId}`);

        chatEventSource.onopen = () => console.log("Chat SSE connection opened.");

        chatEventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'audio_response_data') { return; }
            if (data.type === 'tool_attempt' || data.type === 'tool_result') { return; }
            addChatMessage(data.data, data.type);
        };

        chatEventSource.onerror = (error) => {
            console.error('Chat EventSource failed:', error);
            addChatMessage('Connection to chat server lost. Please refresh.', 'error');
        };
    }

    async function sendChatMessage(text, audioBlob = null) {
        if (!text && !audioBlob) return;
        if (!chatEventSource || chatEventSource.readyState === EventSource.CLOSED) {
            connectChatEventSource();
        }

        const formData = new FormData();
        if (text) formData.append('text_input', text);
        if (audioBlob) formData.append('audio_file', audioBlob, 'user_audio.webm');

        userInput.value = '';

        try {
            const response = await fetch(`/chat/process_chat/${chatClientId}`, {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) {
                const errorData = await response.json();
                addChatMessage(`Error: ${errorData.detail || response.statusText}`, 'error');
            }
        } catch (error) {
            addChatMessage(`Network Error: ${error.message}`, 'error');
        }
    }

    function addChatMessage(message, type = 'bot') {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${type}-message`);
        let p = document.createElement('p');
        if (typeof message === 'object' && message !== null) {
            p.innerHTML = `<pre>${JSON.stringify(message, null, 2)}</pre>`;
        } else {
            p.innerHTML = (message || "").replace(/\n/g, '<br>');
        }
        messageDiv.appendChild(p);
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // --- Event Listeners for Chat ---
    sendButton.addEventListener('click', () => sendChatMessage(userInput.value.trim()));
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage(userInput.value.trim());
        }
    });

    recordButton.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
            recordButton.textContent = '🎤';
            recordButton.classList.remove('recording');
        } else {
            navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    sendChatMessage(null, audioBlob);
                    stream.getTracks().forEach(track => track.stop());
                };
                mediaRecorder.start();
                recordButton.textContent = '⏹️';
                recordButton.classList.add('recording');
            }).catch(err => addChatMessage(`Mic error: ${err.message}`, 'error'));
        }
    });

    // --- Modal Close ---
    closeModal.addEventListener('click', () => modalBackdrop.style.display = 'none');
    modalBackdrop.addEventListener('click', (e) => {
        if (e.target === modalBackdrop) modalBackdrop.style.display = 'none';
    });

    // --- INITIALIZATION ---
    loadInstructions();
    loadAllTestCases();
    connectChatEventSource();
});