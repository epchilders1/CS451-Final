import MagicBento from "../components/MagicBento";
import PixelBlast from "../components/PixelBlast";
import React from "react";
import { useEffect, useState } from "react";

export default function HomePage() {
    const [predictionData, setPredictionData] = useState(null);

    const summaryAPICall = async () => {

        try {
            const response = await fetch(
            'https://7dl05fuc73.execute-api.us-east-2.amazonaws.com/default/netflix-engagement-predictor',
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({})
            }
            );

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setPredictionData(data);
            console.log('Fetched summary data:', data);
        }
        catch (error) {
            console.error('Error fetching summary:', error);
        }
    }

    useEffect(() => {
        summaryAPICall();
    }, []);
    return(
        <div style={{ 
            width: '100vw', 
            height: '100vh', 
            position: 'relative',
            overflow: 'hidden',
            background: 'radial-gradient(ellipse at top, #1a0b2e 0%, #0a0014 50%, #000000 100%)'
        }}>
            <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                zIndex: 0
            }}>
                <PixelBlast
                    variant="circle"
                    pixelSize={6}
                    color="#B19EEF"
                    patternScale={3}
                    patternDensity={1.2}
                    pixelSizeJitter={0.5}
                    enableRipples
                    rippleSpeed={0.4}
                    rippleThickness={0.12}
                    rippleIntensityScale={1.5}
                    liquid
                    liquidStrength={0.12}
                    liquidRadius={1.2}
                    liquidWobbleSpeed={5}
                    speed={0.6}
                    edgeFade={0.25}
                    transparent
                />
            </div>

            <div style={{
                position: 'relative',
                zIndex: 10,
                width: '100%',
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
            }}>
                <MagicBento 
                    textAutoHide={true}
                    enableStars={true}
                    enableSpotlight={true}
                    enableBorderGlow={true}
                    enableTilt={true}
                    enableMagnetism={true}
                    clickEffect={true}
                    spotlightRadius={300}
                    particleCount={12}
                    glowColor="132, 0, 255"
                />
            </div>
        </div>
    );
}