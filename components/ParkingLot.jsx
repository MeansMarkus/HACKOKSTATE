import { useParkingSpots } from '../hooks/useParkingSpots';
import { useEffect } from 'react';

function ParkingLot() {
  const { spots, initializeSpots } = useParkingSpots();

  // Initialize spots when component mounts
  useEffect(() => {
    if (spots.length === 0) {
      initializeSpots(20);
    }
  }, [spots.length, initializeSpots]);

  if (spots.length === 0) {
    return <div>Loading parking spots...</div>;
  }

  return (
    <div className="parking-lot">
      <h2>Available Spots: {spots.filter(s => !s.occupied).length}</h2>
      <div className="spots-grid">
        {spots.map(spot => (
          <div 
            key={spot.id}
            className={`spot ${spot.occupied ? 'occupied' : 'available'}`}
          >
            <span>{spot.number}</span>
            {spot.occupied ? '🚗' : '🅿️'}
          </div>
        ))}
      </div>
    </div>
  );
}