import MagicBento from "../components/MagicBento";
import PixelBlast from "../components/PixelBlast";
import React, { use } from "react";
import { useEffect, useState } from "react";
import Loading from "../components/Loading";

export default function HomePage() {
    const [predictionData, setPredictionData] = useState(null);

    const summaryAPICall = async () => {

        try {
            const response = await fetch(
                "https://7dl05fuc73.execute-api.us-east-2.amazonaws.com/default/netflix-engagement-predictor",
                {
                    method: "GET"
                }
                );


            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setPredictionData(JSON.parse(data.body));
        }
        catch (error) {
            console.error('Error fetching summary:', error);
        }
    }

    useEffect(() => {
        summaryAPICall();
    }, []);

    useEffect(() => {
        if (predictionData) {
            console.log("Prediction Data:", predictionData);
        }
    }, [predictionData]);

    if(!predictionData) {
        return <Loading />;
    }
    else{
    return(
        <div style={{ 
            width: '100vw', 
            height: '100vh', 
            position: 'relative',
            overflow: 'auto',
            background: 'radial-gradient(ellipse at top, #1a0b2e 0%, #0a0014 50%, #000000 100%)',
            zIndex: 0
        }}>
                <div style={{
                position: "fixed",
                top: 0,
                left: 0,
                width: "100vw",
                height: "100vh",
                zIndex: -1,
                pointerEvents: "none"
                }}>

                <PixelBlast
                    pixelSize={4}
                    patternDensity={1}
                    edgeFade={0}
                    pixelSizeJitter={0}
                    noiseAmount={0.4}
                />
            </div>
              <div style={{
                position: 'relative',
                zIndex: 10,
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'auto',
                flexDirection: 'column',
                padding: '30px 20px 40px 20px', 
                color: '#fff',
                backgroundColor: 'rgba(0, 0, 0, 0.4)', 
            }}>
                <h1 
                    style={{
                        fontSize: '48px',
                        fontWeight: '800',
                        margin: '0',
                    }}
                >
                    Netflix Engagement Predictor
                </h1>
                <p 
                    style={{
                        fontSize: '18px',
                        fontWeight: '400',
                        color: '#e0e0e0',
                        maxWidth: '700px', 
                        textAlign: 'center',
                        marginTop: '10px',
                        marginBottom: '0',
                        lineHeight: '1.4',
                        overflowWrap: 'break-word',
                        overflow: 'auto'
                    }}
                >
                    This dashboard automatically updates every week and predicts whether or not the following week will see a downturn in user engagement
                </p>

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
                    predictionData={predictionData}
                />
            </div>
        </div>
    );
    }
}