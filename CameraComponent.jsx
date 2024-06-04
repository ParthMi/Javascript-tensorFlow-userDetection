

import React, { useRef, useEffect, useState } from 'react';
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
import Webcam from "react-webcam";

const CameraComponent = () => {
    const webcamRef = useRef(null);
    const canvasRef = useRef(null);
    const requestRef = useRef(null);
    const [userframe, setUserframe] = useState(false);

    const runPosenet = async () => {
        await tf.setBackend('webgl');
        await tf.ready();

        const net = await posenet.load({
            inputResolution: { width: 640, height: 480 },
            scale: 0.8,
        });

        const detect = async () => {
            if (
                typeof webcamRef.current !== "undefined" &&
                webcamRef.current !== null &&
                webcamRef.current.video.readyState === 4
            ) {
                const video = webcamRef.current.video;
                const videoWidth = webcamRef.current.video.videoWidth;
                const videoHeight = webcamRef.current.video.videoHeight;

                webcamRef.current.video.width = videoWidth;
                webcamRef.current.video.height = videoHeight;

                const pose = await net.estimateSinglePose(video);
                const minConfidence = 0.5;
                const keyPoints = pose?.keypoints.filter((kp) => kp.score > minConfidence);
                const userInFrame = keyPoints.length > 0;
                setUserframe(userInFrame);
                
            }

            requestRef.current = requestAnimationFrame(detect);
        };

        detect();
    };
    useEffect(() => {
        runPosenet();

        return () => {
            cancelAnimationFrame(requestRef.current);
        };
    }, []);

    return (
        <div className="container">

            <div className="camera-container">
                <Webcam
                    mirrored
                    ref={webcamRef}
                    className="webcam"
                />
                <canvas
                    ref={canvasRef}
                    className="canvas"
                />

            </div>
            <div className='content'>{userframe.toString()}</div>
        </div>
    );
}

export default CameraComponent;
