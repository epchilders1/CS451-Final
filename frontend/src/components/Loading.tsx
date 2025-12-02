import PixelBlast from "../components/PixelBlast";
import ShinyText from "../components/ShinyText";

export default function Loading() {
    return (
        <div style={{ 
            width: '100vw', 
            height: '100vh', 
            position: 'relative',
            overflow: 'hidden',
            background: 'radial-gradient(ellipse at top, #1a0b2e 0%, #0a0014 50%, #000000 100%)',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            gap: '2rem' // space between text and loader
        }}>
            {/* Background effect */}
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
                    patternDensity={1.4}
                    pixelSizeJitter={0.5}
                    enableRipples
                    rippleSpeed={0.6}
                    rippleThickness={0.12}
                    rippleIntensityScale={1.5}
                    liquid
                    liquidStrength={0.12}
                    liquidRadius={1.2}
                    liquidWobbleSpeed={5}
                    speed={1.5}
                    edgeFade={0.25}
                    transparent
                />
            </div>

            <div style={{ zIndex: 1, textAlign: 'center', backgroundColor: 'rgba(0, 0, 0, 0.5)', padding: '1rem', borderRadius: '10px' }}>
                <ShinyText 
                    text="Loading Netflix Engagement Prediction..." 
                    disabled={false} 
                    speed={3} 
                    className='custom-class'
                    style={{ fontSize: '3rem' }}
                />
            </div>
        </div>
    );
}
