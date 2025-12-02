import './ShinyText.css';

export default function ShinyText(props:any){
    const {text, disabled, speed, className} = props;
  const animationDuration = `${speed}s`;

  return (
    <div className={`shiny-text ${disabled ? 'disabled' : ''} ${className}`} style={{ animationDuration }}>
      {text}
    </div>
  );
};


