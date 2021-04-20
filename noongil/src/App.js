import './App.css';
import React, { useEffect, useRef } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';

function App() {

    /**
     * voice recognition
     */
    const commands = [
        {
            command: 'Can you read this for me',
            callback: () => alert('You just said hi!!')
        },
    ]
    const { transcript } = useSpeechRecognition({ commands })

    /**
     * camera feed
     */
    const videoRef = useRef(null);

    const getVideo = () => {
        navigator.mediaDevices
            .getUserMedia({ video: { width: 300 } })
            .then(stream => {
                let video = videoRef.current;
                video.srcObject = stream;
                video.play();
            })
            .catch(err => {
                console.error("error:", err);
            });
    };

    useEffect(() => {
        if (!SpeechRecognition.browserSupportsSpeechRecognition()) {
            alert("Ups, your browser is not supported!");
        }
        getVideo();
    }, [videoRef]);

    return (
        <div>
            <h3>Noongil</h3>
            <p>{transcript ? transcript : 'Press the button to start listening'}</p>

            <button onClick={SpeechRecognition.startListening}>Start listening</button>
            &nbsp;
            <button onClick={SpeechRecognition.stopListening}>Stop listening</button>

            <button>Take a photo</button>
            <video ref={videoRef}/>
            <canvas />
      </div>
    );
}

export default App;
