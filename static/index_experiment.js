
        document.addEventListener('DOMContentLoaded', function() {
            // Theme toggle (existing, but ensuring it's here if needed)
            const themeToggle = document.getElementById('theme-toggle');
            themeToggle.addEventListener('click', () => {
                document.body.classList.toggle('dark');
            });

            // Start transcription
            const startButton = document.getElementById('start-transcription');
            const stopButton = document.getElementById('stop-transcription');
            const status = document.getElementById('transcription-status');
            const transcriptElement = document.getElementById('transcript');
            const llmResponseElement = document.getElementById('llm-response');

            startButton.addEventListener('click', function() {
                fetch('/start_transcription', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'Transcription started') {
                            status.textContent = 'Active';
                            startButton.disabled = true;
                            stopButton.disabled = false;
                            startPolling();
                        } else {
                            alert(data.status);
                        }
                    });
            });

            // Stop transcription
            stopButton.addEventListener('click', function() {
                fetch('/stop_transcription', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        status.textContent = 'Idle';
                        startButton.disabled = false;
                        stopButton.disabled = true;
                        stopPolling();
                    });
            });

            // Polling for data (to update transcript and LLM response in real-time)
            let pollingInterval;
            function startPolling() {
                pollingInterval = setInterval(() => {
                    fetch('/get_data')
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'Active') {
                                transcriptElement.textContent = data.transcript || 'No transcript available yet.';
                                llmResponseElement.textContent = data.llm_response || 'No response yet.';
                            } else {
                                transcriptElement.textContent = 'Transcription inactive.';
                                llmResponseElement.textContent = 'No response yet.';
                            }
                        });
                }, 2000); // Poll every 2 seconds
            }

            function stopPolling() {
                clearInterval(pollingInterval);
            }

            

            // Refresh documents
            const refreshDocuments = document.getElementById('refresh-documents');
            const documentsTable = document.getElementById('documents-table');

            refreshDocuments.addEventListener('click', function() {
                fetch('/get_documents')
                    .then(response => response.json())
                    .then(documents => {
                        documentsTable.innerHTML = '';
                        documents.forEach(doc => {
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td class="border p-2">${doc.filename}</td>
                                <td class="border p-2">${doc.summary}</td>
                                <td class="border p-2">${new Date(doc.upload_time * 1000).toLocaleString()}</td>
                            `;
                            documentsTable.appendChild(row);
                        });
                    });
            });

            // Update chunking config
            const updateChunking = document.getElementById('update-chunking');
            updateChunking.addEventListener('click', function() {
                const data = {
                    chunk_size_ingest: document.getElementById('chunk-size-ingest').value,
                    chunk_overlap_ingest: document.getElementById('chunk-overlap-ingest').value,
                    chunk_size_llm: document.getElementById('chunk-size-llm').value,
                    chunk_overlap_llm: document.getElementById('chunk-overlap-llm').value,
                    similarity_threshold: document.getElementById('similarity-threshold').value
                };
                fetch('/set_chunking_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                })
                .then(response => response.json())
                .then(result => {
                    alert(result.status);
                });
            });

            // Last retrieval and chunking info (poll or refresh)
            const lastRetrieval = document.getElementById('last-retrieval');
            const chunkingInfo = document.getElementById('chunking-info');

            function refreshInfo() {
                fetch('/get_last_retrieval')
                    .then(response => response.json())
                    .then(data => {
                        lastRetrieval.textContent = JSON.stringify(data, null, 2);
                    });

                fetch('/get_chunking_info')
                    .then(response => response.json())
                    .then(data => {
                        chunkingInfo.textContent = JSON.stringify(data, null, 2);
                    });
            }

            // Call refreshInfo periodically or on events
            setInterval(refreshInfo, 5000); // Every 5 seconds

            

            
        });
    