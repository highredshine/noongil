import './App.css';
import { useEffect } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';

function App() {
    const commands = [
        {
            command: 'Can you read this for me',
            callback: () => alert('You just said hi!!')
        },
    ]
    const { transcript } = useSpeechRecognition({ commands })

    useEffect(() => {
        if (!SpeechRecognition.browserSupportsSpeechRecognition()) {
            alert("Ups, your browser is not supported!");
        }
    }, []);

    return (
        <div>
            <h3>Noongil</h3>
            <p>{transcript ? transcript : 'Press the button to start listening'}</p>

            <button onClick={SpeechRecognition.startListening}>Start listening</button>
            &nbsp;
            <button onClick={SpeechRecognition.stopListening}>Stop listening</button>
      </div>
    );
}

export default App;
