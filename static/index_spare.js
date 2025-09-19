 // Weather Widget
        function updateWeather() {
            fetch('/weather').then(res => res.json()).then(data => {
                document.getElementById('weatherDisplay').textContent = data.weather || '25°C, Sunny';
            }).catch(() => {
                document.getElementById('weatherDisplay').textContent = '25°C, Sunny';
            });
        }


// Clock
        function startTime() {
            const today = new Date();
            let h = today.getHours();
            let m = today.getMinutes();
            let s = today.getSeconds();
            m = m < 10 ? "0" + m : m;
            s = s < 10 ? "0" + s : s;
            document.getElementById('clock').textContent = h + ":" + m + ":" + s;
            setTimeout(startTime, 1000);
        }


// Start/Stop Transcription
        function startTranscription() {
            document.getElementById('startIcon').classList.add('animate-spin');
            fetch('/start_transcription', {method: 'POST'}).then(res => res.json()).then(data => {
                document.getElementById('transcriberStatus').textContent = data.status || 'Active';
            });
        }
        function stopTranscription() {
            document.getElementById('startIcon').classList.remove('animate-spin');
            fetch('/stop_transcription', {method: 'POST'}).then(res => res.json()).then(data => {
                document.getElementById('transcriberStatus').textContent = data.status || 'Idle';
            });
        }
