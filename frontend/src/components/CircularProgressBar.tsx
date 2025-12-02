import * as React from 'react';
import Stack from '@mui/joy/Stack';
import Typography from '@mui/joy/Typography';
import CircularProgress from '@mui/joy/CircularProgress';
// Assuming 'use-count-up' is available
import { useCountUp } from 'use-count-up'; 
import { useState, useEffect } from 'react';

// Define the shape of the props for better TypeScript integration
interface CircularProgressBarProps {
    desiredValue: number;
    loading: boolean;
    activeTab: boolean;
    ratingView: boolean;
}

// Define the custom glow colors
const GLOW_COLOR = '#D8B4FE'; // Light purple for the fill and glow
const BACKGROUND_PURPLE = '#2A0548'; // Darker purple for the track/background

export default function CircularProgressBar(props: CircularProgressBarProps) {
    const { desiredValue, loading, activeTab, ratingView } = props;
    
    // Note: The original file had a redundant 'buttonLabel' state which is removed.

    const { value } = useCountUp({
        isCounting: loading && activeTab,
        duration: 1, // Animation duration in seconds
        start: 0,
        end: desiredValue,
    });

    // Ensure the value is treated as a number
    const numericValue = Number(value);

    return (
        <Stack
            direction="row"
            spacing={8}
            sx={{ 
                alignItems: 'center', 
                flexWrap: 'wrap',
                // Optional: Add a subtle outer glow to the entire component area
                // border: '1px solid rgba(216, 180, 254, 0.1)',
                // borderRadius: '16px',
            }}
            className="flex justify-center items-center"
        >
            <Stack spacing={2}>
                <CircularProgress
                    size="md"
                    determinate
                    value={numericValue}
                    // Apply custom styles for the glowy purple effect
                    sx={{
                        '--CircularProgress-size': '150px', // Larger size to match the image
                        '--CircularProgress-trackThickness': '20px', // Thicker track
                        '--CircularProgress-progressThickness': '20px', // Thicker progress stroke
                        
                        // 1. Color the track a dark purple
                        '--CircularProgress-trackColor': BACKGROUND_PURPLE,
                        
                        // 2. FIX: Use the explicit CSS variable to color the progress stroke
                        '--CircularProgress-progressColor': GLOW_COLOR,

                        // 3. Add a glowing shadow to the progress stroke
                        '& .MuiCircularProgress-progress': {
                            // The glow filter remains correct
                            filter: 'drop-shadow(0 0 3px #D8B4FE) drop-shadow(0 0 5px #C084FC)',
                        },
                        
                        // 4. Ensure the root element doesn't interfere with the glow
                        backgroundColor: 'transparent',
                    }}
                >
                    <Typography
                        level="h1"
                        sx={{
                            fontSize: '2.5rem',
                            fontWeight: 'bold',
                            color: 'white', // Ensure text is white for contrast
                            textShadow: `0 0 10px ${GLOW_COLOR}`, // Match text glow to the bar
                        }}
                    >
                        {ratingView 
                            ? `${Math.round(numericValue / 10)}/10` 
                            : `${Math.round(numericValue)}%`
                        }
                    </Typography>
                </CircularProgress>
            </Stack>
        </Stack>
    );
}