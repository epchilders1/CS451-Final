import React, { useRef, useEffect, useCallback, useState } from 'react';
import { gsap } from 'gsap';
import CircularProgressBar from './CircularProgressBar';

// Types
interface BentoCardProps {
  color?: string;
  title?: string;
  description?: string;
  label?: string;
}

interface BentoProps {
  textAutoHide?: boolean;
  enableStars?: boolean;
  enableSpotlight?: boolean;
  enableBorderGlow?: boolean;
  disableAnimations?: boolean;
  spotlightRadius?: number;
  particleCount?: number;
  enableTilt?: boolean;
  glowColor?: string;
  clickEffect?: boolean;
  enableMagnetism?: boolean;
  predictionData?: any;
}

const DEFAULT_PARTICLE_COUNT = 12;
const DEFAULT_SPOTLIGHT_RADIUS = 300;
const DEFAULT_GLOW_COLOR = '132, 0, 255';
const MOBILE_BREAKPOINT = 768;

// Inline styles object
const styles = `
  .bento-container {
    min-height: 100vh;
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
  }

  .bento-container * {
    box-sizing: border-box;
  }
  
  .card-grid {
    display: grid;
    gap: 1.5rem;
    width: 100%;
    max-width: 1400px;
    font-size: clamp(1rem, 0.9rem + 0.5vw, 1.5rem);
  }

  .magic-bento-card {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    position: relative;
    aspect-ratio: 4/3;
    min-height: 250px;
    width: 100%;
    max-width: 100%;
    padding: 2rem;
    border-radius: 24px;
    border: 1px solid #392e4e;
    background: #060010;
    font-weight: 300;
    overflow: hidden;
    transition: all 0.3s ease;
    cursor: pointer;
    color: white;
  }

  .magic-bento-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
  }

  .magic-bento-card__header {
    display: flex;
    gap: 0.75em;
    justify-content: space-between;
    position: relative;
    color: white;
  }

  .magic-bento-card__content {
    display: flex;
    flex-direction: column;
    position: relative;
    color: white;
  }

  .magic-bento-card__label {
    font-size: 18px;
    opacity: 0.8;
  }

  .magic-bento-card__title {
    font-weight: 400;
    font-size: 24px;
    margin: 0 0 0.5rem;
  }

  .magic-bento-card__description {
    font-size: 16px;
    line-height: 1.4;
    opacity: 0.9;
    margin: 0;
  }

  .magic-bento-card--text-autohide .magic-bento-card__title,
  .magic-bento-card--text-autohide .magic-bento-card__description {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .magic-bento-card--text-autohide .magic-bento-card__title {
    -webkit-line-clamp: 1;
    line-clamp: 1;
  }

  .magic-bento-card--text-autohide .magic-bento-card__description {
    -webkit-line-clamp: 2;
    line-clamp: 2;
  }

  @media (max-width: 599px) {
    .card-grid {
      grid-template-columns: 1fr;
      width: 90%;
      margin: 0 auto;
      padding: 0.5em;
    }

    .magic-bento-card {
      width: 100%;
      min-height: 180px;
    }
  }

  @media (min-width: 600px) {
    .card-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  @media (min-width: 1024px) {
    .card-grid {
      grid-template-columns: repeat(4, 1fr);
    }

    .magic-bento-card:nth-child(3) {
      grid-column: span 2;
      grid-row: span 2;
    }

    .magic-bento-card:nth-child(4) {
      grid-column: 1 / span 2;
      grid-row: 2 / span 2;
    }

    .magic-bento-card:nth-child(6) {
      grid-column: 4;
      grid-row: 3;
    }
  }

  .magic-bento-card--border-glow::after {
    content: '';
    position: absolute;
    inset: 0;
    padding: 6px;
    background: radial-gradient(
      var(--glow-radius, 200px) circle at var(--glow-x, 50%) var(--glow-y, 50%),
      rgba(132, 0, 255, calc(var(--glow-intensity, 0) * 0.8)) 0%,
      rgba(132, 0, 255, calc(var(--glow-intensity, 0) * 0.4)) 30%,
      transparent 60%
    );
    border-radius: inherit;
    -webkit-mask:
      linear-gradient(#fff 0 0) content-box,
      linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask:
      linear-gradient(#fff 0 0) content-box,
      linear-gradient(#fff 0 0);
    mask-composite: exclude;
    pointer-events: none;
    opacity: 1;
    transition: opacity 0.3s ease;
    z-index: 1;
  }

  .magic-bento-card--border-glow:hover {
    box-shadow:
      0 4px 20px rgba(46, 24, 78, 0.4),
      0 0 30px rgba(132, 0, 255, 0.2);
  }

  .particle-container {
    position: relative;
    overflow: hidden;
  }

  .global-spotlight {
    mix-blend-mode: screen;
    will-change: transform, opacity;
    z-index: 200 !important;
    pointer-events: none;
  }

  .bento-section {
    position: relative;
    user-select: none;
  }
`;

const createParticleElement = (x: number, y: number, color: string): HTMLDivElement => {
  const el = document.createElement('div');
  el.className = 'particle';
  el.style.cssText = `
    position: absolute;
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: rgba(${color}, 1);
    box-shadow: 0 0 6px rgba(${color}, 0.6);
    pointer-events: none;
    z-index: 100;
    left: ${x}px;
    top: ${y}px;
  `;
  return el;
};

const calculateSpotlightValues = (radius: number) => ({
  proximity: radius * 0.5,
  fadeDistance: radius * 0.75
});

const updateCardGlowProperties = (
  card: HTMLElement,
  mouseX: number,
  mouseY: number,
  glow: number,
  radius: number
) => {
  const rect = card.getBoundingClientRect();
  const relativeX = ((mouseX - rect.left) / rect.width) * 100;
  const relativeY = ((mouseY - rect.top) / rect.height) * 100;

  card.style.setProperty('--glow-x', `${relativeX}%`);
  card.style.setProperty('--glow-y', `${relativeY}%`);
  card.style.setProperty('--glow-intensity', glow.toString());
  card.style.setProperty('--glow-radius', `${radius}px`);
};

const ParticleCard: React.FC<{
  children: React.ReactNode;
  className?: string;
  disableAnimations?: boolean;
  style?: React.CSSProperties;
  particleCount?: number;
  glowColor?: string;
  enableTilt?: boolean;
  clickEffect?: boolean;
  enableMagnetism?: boolean;
}> = ({
  children,
  className = '',
  disableAnimations = false,
  style,
  particleCount = DEFAULT_PARTICLE_COUNT,
  glowColor = DEFAULT_GLOW_COLOR,
  enableTilt = true,
  clickEffect = false,
  enableMagnetism = false
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const particlesRef = useRef<HTMLDivElement[]>([]);
  const timeoutsRef = useRef<number[]>([]);
  const isHoveredRef = useRef(false);
  const memoizedParticles = useRef<HTMLDivElement[]>([]);
  const particlesInitialized = useRef(false);

  const initializeParticles = useCallback(() => {
    if (particlesInitialized.current || !cardRef.current) return;

    const { width, height } = cardRef.current.getBoundingClientRect();
    memoizedParticles.current = Array.from({ length: particleCount }, () =>
      createParticleElement(Math.random() * width, Math.random() * height, glowColor)
    );
    particlesInitialized.current = true;
  }, [particleCount, glowColor]);

  const clearAllParticles = useCallback(() => {
    timeoutsRef.current.forEach(clearTimeout);
    timeoutsRef.current = [];

    particlesRef.current.forEach(particle => {
      gsap.to(particle, {
        scale: 0,
        opacity: 0,
        duration: 0.3,
        ease: 'back.in(1.7)',
        onComplete: () => {
          particle.parentNode?.removeChild(particle);
        }
      });
    });
    particlesRef.current = [];
  }, []);

  const animateParticles = useCallback(() => {
    if (!cardRef.current || !isHoveredRef.current) return;

    if (!particlesInitialized.current) {
      initializeParticles();
    }

    memoizedParticles.current.forEach((particle, index) => {
      const timeoutId = window.setTimeout(() => {
        if (!isHoveredRef.current || !cardRef.current) return;

        const clone = particle.cloneNode(true) as HTMLDivElement;
        cardRef.current.appendChild(clone);
        particlesRef.current.push(clone);

        gsap.fromTo(clone, { scale: 0, opacity: 0 }, { scale: 1, opacity: 1, duration: 0.3, ease: 'back.out(1.7)' });

        gsap.to(clone, {
          x: (Math.random() - 0.5) * 100,
          y: (Math.random() - 0.5) * 100,
          rotation: Math.random() * 360,
          duration: 2 + Math.random() * 2,
          ease: 'none',
          repeat: -1,
          yoyo: true
        });

        gsap.to(clone, {
          opacity: 0.3,
          duration: 1.5,
          ease: 'power2.inOut',
          repeat: -1,
          yoyo: true
        });
      }, index * 100);

      timeoutsRef.current.push(timeoutId);
    });
  }, [initializeParticles]);

  useEffect(() => {
    if (disableAnimations || !cardRef.current) return;

    const element = cardRef.current;

    const handleMouseEnter = () => {
      isHoveredRef.current = true;
      animateParticles();

      if (enableTilt) {
        gsap.to(element, {
          rotateX: 5,
          rotateY: 5,
          duration: 0.3,
          ease: 'power2.out',
          transformPerspective: 1000
        });
      }
    };

    const handleMouseLeave = () => {
      isHoveredRef.current = false;
      clearAllParticles();

      if (enableTilt) {
        gsap.to(element, {
          rotateX: 0,
          rotateY: 0,
          duration: 0.3,
          ease: 'power2.out'
        });
      }

      if (enableMagnetism) {
        gsap.to(element, {
          x: 0,
          y: 0,
          duration: 0.3,
          ease: 'power2.out'
        });
      }
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!enableTilt && !enableMagnetism) return;

      const rect = element.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;

      if (enableTilt) {
        const rotateX = ((y - centerY) / centerY) * -10;
        const rotateY = ((x - centerX) / centerX) * 10;

        gsap.to(element, {
          rotateX,
          rotateY,
          duration: 0.1,
          ease: 'power2.out',
          transformPerspective: 1000
        });
      }

      if (enableMagnetism) {
        const magnetX = (x - centerX) * 0.05;
        const magnetY = (y - centerY) * 0.05;

        gsap.to(element, {
          x: magnetX,
          y: magnetY,
          duration: 0.3,
          ease: 'power2.out'
        });
      }
    };

    const handleClick = (e: MouseEvent) => {
      if (!clickEffect) return;

      const rect = element.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const maxDistance = Math.max(
        Math.hypot(x, y),
        Math.hypot(x - rect.width, y),
        Math.hypot(x, y - rect.height),
        Math.hypot(x - rect.width, y - rect.height)
      );

      const ripple = document.createElement('div');
      ripple.style.cssText = `
        position: absolute;
        width: ${maxDistance * 2}px;
        height: ${maxDistance * 2}px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(${glowColor}, 0.4) 0%, rgba(${glowColor}, 0.2) 30%, transparent 70%);
        left: ${x - maxDistance}px;
        top: ${y - maxDistance}px;
        pointer-events: none;
        z-index: 1000;
      `;

      element.appendChild(ripple);

      gsap.fromTo(
        ripple,
        { scale: 0, opacity: 1 },
        {
          scale: 1,
          opacity: 0,
          duration: 0.8,
          ease: 'power2.out',
          onComplete: () => ripple.remove()
        }
      );
    };

    element.addEventListener('mouseenter', handleMouseEnter);
    element.addEventListener('mouseleave', handleMouseLeave);
    element.addEventListener('mousemove', handleMouseMove);
    element.addEventListener('click', handleClick);

    return () => {
      isHoveredRef.current = false;
      element.removeEventListener('mouseenter', handleMouseEnter);
      element.removeEventListener('mouseleave', handleMouseLeave);
      element.removeEventListener('mousemove', handleMouseMove);
      element.removeEventListener('click', handleClick);
      clearAllParticles();
    };
  }, [animateParticles, clearAllParticles, disableAnimations, enableTilt, enableMagnetism, clickEffect, glowColor]);

  return (
    <div
      ref={cardRef}
      className={`${className} particle-container`}
      style={{ ...style, position: 'relative', overflow: 'hidden' }}
    >
      {children}
    </div>
  );
};
const getMedalColor = (index: number) => {
  switch (index) {
    case 0:
      return '#FFD700'; 
    case 1:
      return '#C0C0C0'; 
    case 2: 
      return '#CD7F32';
    default:
      return '#e0e0e0'; 
  }
};


const findBestModel = (data: { [model: string]: any }) => {
    let bestModel: string | null = null;
    let maxAuc = -1;
    for (const [model, metrics] of Object.entries(data)) {
        if (metrics.cv_auc_mean > maxAuc) {
            maxAuc = metrics.cv_auc_mean;
            bestModel = model;
        }
    }
    return bestModel;
};

const formatAuc = (value:any) => value.toFixed(3);

const formatStd = (value:any) => `±${value.toFixed(3)}`;

const GlobalSpotlight: React.FC<{
  gridRef: React.RefObject<HTMLDivElement | null>;
  disableAnimations?: boolean;
  enabled?: boolean;
  spotlightRadius?: number;
  glowColor?: string;
}> = ({
  gridRef,
  disableAnimations = false,
  enabled = true,
  spotlightRadius = DEFAULT_SPOTLIGHT_RADIUS,
  glowColor = DEFAULT_GLOW_COLOR
}) => {
  const spotlightRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (disableAnimations || !gridRef?.current || !enabled) return;

    const spotlight = document.createElement('div');
    spotlight.className = 'global-spotlight';
    spotlight.style.cssText = `
      position: fixed;
      width: 800px;
      height: 800px;
      border-radius: 50%;
      pointer-events: none;
      background: radial-gradient(circle,
        rgba(${glowColor}, 0.15) 0%,
        rgba(${glowColor}, 0.08) 15%,
        rgba(${glowColor}, 0.04) 25%,
        rgba(${glowColor}, 0.02) 40%,
        rgba(${glowColor}, 0.01) 65%,
        transparent 70%
      );
      z-index: 200;
      opacity: 0;
      transform: translate(-50%, -50%);
      mix-blend-mode: screen;
    `;
    document.body.appendChild(spotlight);
    spotlightRef.current = spotlight;

    const handleMouseMove = (e: MouseEvent) => {
      if (!spotlightRef.current || !gridRef.current) return;

      const section = gridRef.current.closest('.bento-section');
      const rect = section?.getBoundingClientRect();
      const mouseInside =
        rect && e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom;

      const cards = gridRef.current.querySelectorAll('.magic-bento-card');

      if (!mouseInside) {
        gsap.to(spotlightRef.current, {
          opacity: 0,
          duration: 0.3,
          ease: 'power2.out'
        });
        cards.forEach(card => {
          (card as HTMLElement).style.setProperty('--glow-intensity', '0');
        });
        return;
      }

      const { proximity, fadeDistance } = calculateSpotlightValues(spotlightRadius);
      let minDistance = Infinity;

      cards.forEach(card => {
        const cardElement = card as HTMLElement;
        const cardRect = cardElement.getBoundingClientRect();
        const centerX = cardRect.left + cardRect.width / 2;
        const centerY = cardRect.top + cardRect.height / 2;
        const distance =
          Math.hypot(e.clientX - centerX, e.clientY - centerY) - Math.max(cardRect.width, cardRect.height) / 2;
        const effectiveDistance = Math.max(0, distance);

        minDistance = Math.min(minDistance, effectiveDistance);

        let glowIntensity = 0;
        if (effectiveDistance <= proximity) {
          glowIntensity = 1;
        } else if (effectiveDistance <= fadeDistance) {
          glowIntensity = (fadeDistance - effectiveDistance) / (fadeDistance - proximity);
        }

        updateCardGlowProperties(cardElement, e.clientX, e.clientY, glowIntensity, spotlightRadius);
      });

      gsap.to(spotlightRef.current, {
        left: e.clientX,
        top: e.clientY,
        duration: 0.1,
        ease: 'power2.out'
      });

      const targetOpacity =
        minDistance <= proximity
          ? 0.8
          : minDistance <= fadeDistance
            ? ((fadeDistance - minDistance) / (fadeDistance - proximity)) * 0.8
            : 0;

      gsap.to(spotlightRef.current, {
        opacity: targetOpacity,
        duration: targetOpacity > 0 ? 0.2 : 0.5,
        ease: 'power2.out'
      });
    };

    document.addEventListener('mousemove', handleMouseMove);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      spotlightRef.current?.parentNode?.removeChild(spotlightRef.current);
    };
  }, [gridRef, disableAnimations, enabled, spotlightRadius, glowColor]);

  return null;
};

const useMobileDetection = () => {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth <= MOBILE_BREAKPOINT);

    checkMobile();
    window.addEventListener('resize', checkMobile);

    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  return isMobile;
};

const MagicBento: React.FC<BentoProps> = ({
  textAutoHide = true,
  enableStars = true,
  enableSpotlight = true,
  enableBorderGlow = true,
  disableAnimations = false,
  spotlightRadius = DEFAULT_SPOTLIGHT_RADIUS,
  particleCount = DEFAULT_PARTICLE_COUNT,
  enableTilt = false,
  glowColor = DEFAULT_GLOW_COLOR,
  clickEffect = true,
  enableMagnetism = true,
  predictionData = null
}) => {
  const gridRef = useRef<HTMLDivElement>(null);
  const isMobile = useMobileDetection();
  const shouldDisableAnimations = disableAnimations || isMobile;

  const baseClassName = `magic-bento-card ${textAutoHide ? 'magic-bento-card--text-autohide' : ''} ${enableBorderGlow ? 'magic-bento-card--border-glow' : ''}`;

  const CardWrapper = enableStars ? ParticleCard : 'div';
  const cardWrapperProps = enableStars ? {
    disableAnimations: shouldDisableAnimations,
    particleCount,
    glowColor,
    enableTilt,
    clickEffect,
    enableMagnetism
  } : {};

  const prediction = predictionData['ensemble_prediction']
  const top10_movies = predictionData['recent_top_10_movies'];
  const top_predictive_features = predictionData['top_predictive_features']['xgboost'] || [];
  const prediction_date = predictionData['prediction_date'];
  const prediction_status = predictionData['status'];
  const model_performance_summary = predictionData['model_performance_summary'];

  return (
    <div className="bento-container">
      <style>{styles}</style>
      
      {enableSpotlight && (
        <GlobalSpotlight
          gridRef={gridRef}
          disableAnimations={shouldDisableAnimations}
          enabled={enableSpotlight}
          spotlightRadius={spotlightRadius}
          glowColor={glowColor}
        />
      )}

      <div className="card-grid bento-section" ref={gridRef}>
        <CardWrapper
          className={baseClassName}
          style={{ backgroundColor: '#060010' }}
          {...cardWrapperProps}
        >
          <div className="magic-bento-card__header">
            <div className="magic-bento-card__label">Engagement Decline Probability</div>
          </div>
          <div className="flex justify-center items-center magic-bento-card__content">
          <CircularProgressBar
            desiredValue={prediction.decline_probability * 100}
            loading={true}
            activeTab={true}
            ratingView={false}
          />
          </div>
        </CardWrapper>

       <CardWrapper
            className={baseClassName}
            style={{
                backgroundColor: '#060010',
                borderRadius: '12px',
                padding: '24px',
                color: '#fff',
            }}
            {...cardWrapperProps}
        >
            <div 
                className="magic-bento-card__header"
                style={{ marginBottom: '16px' }}
            >
                <div className="magic-bento-card__label">
                    Prediction
                </div>
            </div>
            
            <div className="magic-bento-card__content">
                <h2 
                    className="magic-bento-card__title"
                    style={{
                        fontSize: '40px', 
                        fontWeight: '700',
                        lineHeight: '1.1',
                        margin: '0',
                        color: '#fff',
                    }}
                >
                    {prediction.prediction}
                </h2>
                <p 
                    className="magic-bento-card__description"
                    style={{
                        fontSize: '16px',
                        fontWeight: '400',
                        color: '#e0e0e0',
                        marginTop: '12px'
                    }}
                >
                    There is a 
                    <span 
                        style={{
                            fontWeight: '600',
                            color: prediction.risk_level.toLowerCase() === 'low' ? '#8cff8c' : 
                                  prediction.risk_level.toLowerCase() === 'medium' ? '#ffcc00' : 
                                  '#ff4d4d', 
                            margin: '0 4px',
                        }}
                    >
                        {prediction.risk_level}
                    </span> 
                    chance that viewer engagement drops next week.
                </p>
                <p 
                    style={{
                        fontSize: '11px',
                        fontWeight: '400',
                        color: '#a0a0a0',
                        margin: '8px 0 0 0', 
                        padding: '0',
                    }}
                > 
                    {'>10% decline in hours viewed'}
                </p>            
                </div>
          </CardWrapper>

        <CardWrapper
          className={baseClassName}
          style={{ backgroundColor: '#060010' }}
          {...cardWrapperProps}
        >
          <div className="magic-bento-card__header">
            <div className="magic-bento-card__label">Top Predictive Features</div>
          </div>
          {top_predictive_features && top_predictive_features.length > 0 ? (
              <>
                  <div style={{ listStyle: 'none', padding: '0', margin: '0', overflow: 'auto' }} className="overflow-y-auto max-h-64">
                      {top_predictive_features.map((feature: any, index: number) => (
                          <div 
                              className="flex flex-row"
                              key={index}
                              style={{
                                  display: 'flex', 
                                  alignItems: 'baseline',
                                  marginBottom: '8px',
                              }}
                          >
                              <div
                                  style={{
                                      fontSize: index < 3 ? '32px' : '20px',
                                      fontWeight: '700',
                                      width: '30px', 
                                      textAlign: 'right',
                                      marginRight: '12px',
                                      color: getMedalColor(index),
                                      textShadow: index < 3 ? '0 0 4px rgba(255, 255, 255, 0.4)' : 'none', 
                                  }}
                              >
                                  {index + 1}.
                              </div>
                              
                              <p 
                                  style={{
                                      margin: '0',
                                      fontSize: '16px',
                                      color: '#e0e0e0',
                                  }}
                              >
                                  <strong style={{ color: '#fff' }}>
                                      {feature.feature}
                                  </strong> 
                                  &nbsp;
                                  <span style={{ color: '#a0a0a0', fontSize: '14px' }}>
                                      ({feature.importance})
                                  </span>
                              </p>
                          </div>
                      ))}
                  </div>
              </>
          ) : (
              <p style={{ color: '#e0e0e0' }}>No data available</p>
          )}
        </CardWrapper>

       <CardWrapper
          className={baseClassName}
          // Set the overall card styles
          style={{
              backgroundColor: '#060010',
              borderRadius: '12px',
              padding: '24px',
              color: '#fff',
          }}
          {...cardWrapperProps}
      >
          <div 
              className="magic-bento-card__header"
              style={{ marginBottom: '16px' }}
          >
            <div className = "justify-between w-full flex items-center">
              <div 
                  className="magic-bento-card__label"
                  style={{
                      fontSize: '14px',
                      fontWeight: '500',
                      color: '#a0a0a0',
                      letterSpacing: '0.05em',
                      textTransform: 'uppercase'
                  }}
              >
                  Most Recent Top 10 List
              </div>

              <div>
                <p
                 style={{
                      fontSize: '14px',
                      fontWeight: '500',
                      color: '#a0a0a0',
                      letterSpacing: '0.05em',
                      textTransform: 'uppercase'
                  }}>Total Hours Viewed: {predictionData['latest_data_context']['hours_viewed_sum']}</p>
              </div>
              </div>
          </div>
          
          {top10_movies && top10_movies.length > 0 ? (
              <>
                  <div style={{ listStyle: 'none', padding: '0', margin: '0', overflow: 'auto' }} className="overflow-y-auto max-h-64">
                      {top10_movies.map((movie: any, index: number) => (
                          <div 
                              className="flex flex-row"
                              key={index}
                              style={{
                                  display: 'flex', 
                                  alignItems: 'baseline',
                                  marginBottom: '8px',
                              }}
                          >
                              <div
                                  style={{
                                      fontSize: index < 3 ? '32px' : '20px',
                                      fontWeight: '700',
                                      width: '30px', 
                                      textAlign: 'right',
                                      marginRight: '12px',
                                      color: getMedalColor(index),
                                      textShadow: index < 3 ? '0 0 4px rgba(255, 255, 255, 0.4)' : 'none', 
                                  }}
                              >
                                  {movie.ranking}.
                              </div>
                              
                              <p 
                                  style={{
                                      margin: '0',
                                      fontSize: '16px',
                                      color: '#e0e0e0',
                                  }}
                              >
                                  <strong style={{ color: '#fff' }}>
                                      {movie.title}
                                  </strong> 
                                  &nbsp;
                                  <span style={{ color: '#a0a0a0', fontSize: '14px' }}>
                                      ({movie.hours_viewed} hours viewed)
                                  </span>
                              </p>
                          </div>
                      ))}
                  </div>
              </>
          ) : (
              <p style={{ color: '#e0e0e0' }}>No data available</p>
          )}
      </CardWrapper>

       <CardWrapper
            className={baseClassName}
            style={{
                backgroundColor: '#060010',
                borderRadius: '12px',
                padding: '24px',
                color: '#fff',
            }}
            {...cardWrapperProps}
        >
            <div 
                className="magic-bento-card__header"
                style={{ marginBottom: '16px' }}
            >
                <div 
                    className="magic-bento-card__label"
                    style={{
                        fontSize: '14px',
                        fontWeight: '500',
                        color: '#a0a0a0', 
                        letterSpacing: '0.05em',
                        textTransform: 'uppercase'
                    }}
                >
                    Prediction Info
                </div>
            </div>
            
            <div 
                style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '8px 0',
                    borderBottom: '1px solid #1a1a3a',
                }}
            >
                <p 
                    style={{
                        margin: '0',
                        fontSize: '16px',
                        fontWeight: '400',
                        color: '#a0a0a0', 
                    }}
                >
                    Prediction Date:
                </p>
                <p 
                    style={{
                        margin: '0',
                        fontSize: '16px',
                        fontWeight: '600',
                        color: '#fff',
                    }}
                >
                    {prediction_date}
                </p>
            </div>

            <div 
                style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '8px 0',
                }}
            >
                <p 
                    style={{
                        margin: '0',
                        fontSize: '16px',
                        fontWeight: '400',
                        color: '#a0a0a0',
                    }}
                >
                    Status:
                </p>
                <p 
                    style={{
                        margin: '0',
                        fontSize: '16px',
                        fontWeight: '600',
                        color: '#A855F7', 
                        backgroundColor: 'rgba(168, 85, 247, 0.15)',
                        padding: '2px 8px',
                        borderRadius: '6px',
                    }}
                >
                    {prediction_status}
                </p>
            </div>

        </CardWrapper>

       <CardWrapper
            className={baseClassName}
            style={{
                backgroundColor: '#060010',
                borderRadius: '12px',
                padding: '24px',
                color: '#fff',
            }}
            {...cardWrapperProps}
        >
            <div 
                className="magic-bento-card__header"
                style={{ marginBottom: '16px' }}
            >
                <div 
                    className="magic-bento-card__label"
                    style={{
                        fontSize: '14px',
                        fontWeight: '500',
                        color: '#a0a0a0',
                        letterSpacing: '0.05em',
                        textTransform: 'uppercase'
                    }}
                >
                    Model Performance Summary
                </div>
            </div>
            
            <div 
                className="magic-bento-card__content"
                style={{ marginTop: '10px',
                        overflow:'auto'
                }}
                
            >
                <div 
                    style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        padding: '0 0 8px 0',
                        borderBottom: '1px solid #1a1a3a',
                    }}
                >
                    <span style={{ fontSize: '14px', fontWeight: '500', color: '#a0a0a0', width: '40%' }}>Model</span>
                    <span style={{ fontSize: '14px', fontWeight: '500', color: '#a0a0a0', width: '30%', textAlign: 'right' }}>CV AUC Mean</span>
                    <span style={{ fontSize: '14px', fontWeight: '500', color: '#a0a0a0', width: '30%', textAlign: 'right' }}>Std Dev</span>
                </div>

                {Object.entries(model_performance_summary).map(([modelName, metrics], index) => {
                    const isBest = modelName === findBestModel(model_performance_summary);
                    const highlightColor = '#A855F7';
                    const baseTextColor = '#e0e0e0';

                    return (
                        <div 
                            key={modelName}
                            style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                padding: '12px 0',
                                borderBottom: index < Object.keys(model_performance_summary).length - 1 ? '1px solid #0f0f20' : 'none',
                                backgroundColor: isBest ? 'rgba(168, 85, 247, 0.05)' : 'transparent',
                                borderRadius: isBest ? '4px' : '0',
                            }}
                        >
                            <span 
                                style={{
                                    fontSize: '16px',
                                    fontWeight: isBest ? '700' : '600',
                                    color: isBest ? highlightColor : baseTextColor,
                                    width: '40%',
                                    textTransform: 'capitalize',
                                }}
                            >
                                {modelName.replace('_', ' ')} 
                                {isBest && 
                                    <span style={{ 
                                        fontSize: '10px', 
                                        marginLeft: '6px',
                                        padding: '2px 6px',
                                        borderRadius: '4px',
                                        backgroundColor: highlightColor,
                                        color: '#060010', 
                                        fontWeight: '900',
                                    }}>
                                        BEST
                                    </span>
                                }
                            </span>
                            
                            <span 
                                style={{
                                    fontSize: '16px',
                                    fontWeight: isBest ? '700' : '500',
                                    color: isBest ? highlightColor : baseTextColor,
                                    width: '30%',
                                    textAlign: 'right',
                                }}
                            >
                                {formatAuc((metrics as {cv_auc_mean: number}).cv_auc_mean)}
                            </span>
                            
                            <span 
                                style={{
                                    fontSize: '16px',
                                    fontWeight: '400',
                                    color: '#a0a0a0', 
                                    width: '30%',
                                    textAlign: 'right',
                                }}
                            >
                                {formatStd((metrics as { cv_auc_std: number }).cv_auc_std)}
                            </span>
                        </div>
                    );
                })}
            </div>
        </CardWrapper>
      </div>
    </div>
  );
};

export default MagicBento;