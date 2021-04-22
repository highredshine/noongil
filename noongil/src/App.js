import './App.css';
import React, { useState, useEffect, useRef } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import axios from "axios";

function App() {

    /**
     * voice recognition
     */
    const commands = [
        {
            command: 'Can you read this for me',
            callback: () => takePhoto()
        },
    ]
    const { transcript } = useSpeechRecognition({ commands })
    const [output, setOutput] = useState("");

    /**
     * camera feed
     */
    const videoRef = useRef(null);
    const photoRef = useRef(null);

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

    const takePhoto = () => {
        let video = videoRef.current;
        let photo = photoRef.current;
        let ctx = photo.getContext("2d");
        const width = 320;
        const height = 240;
        photo.width = width;
        photo.height = height;
        ctx.drawImage(video, 0, 0, width, height);

        const data = photo.toDataURL("image", 1.0);
        console.log(data)
        const toSend = {"input" : data};
        let config = {
            headers: {
                "Content-Type": "application/json",
                'Access-Control-Allow-Origin': '*',
            }
        }
        axios.post(
            "http://localhost:4567/input",
            toSend,
            config
        )
            .then(response => {
                setOutput(response.data["output"]);
                let msg = new SpeechSynthesisUtterance();
                msg.text = output;
                window.speechSynthesis.speak(msg);
            })
            .catch(function (error) {
                console.log(error);
            });

    };

    /**
     * voice output
     */

    useEffect(() => {
        if (!SpeechRecognition.browserSupportsSpeechRecognition()) {
            alert("Oops, your browser is not supported!");
        }
        SpeechRecognition.startListening({ continuous: true });
        getVideo();
    }, [videoRef]);



    return (
        <div>
            <h3>Noongil</h3>
            <p>{}{transcript ? transcript : 'this part is a temporary transcript of what you say'}</p>
            <video  ref={videoRef} />
            <canvas ref={photoRef} />
            <div>{output}</div>
      </div>
    );
}

export default App;
