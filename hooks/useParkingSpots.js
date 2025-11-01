import { useState, useEffect } from 'react';
import { 
  collection, 
  doc, 
  setDoc, 
  onSnapshot, 
  updateDoc 
} from 'firebase/firestore';
import { db } from '../firebase-config';

export const useParkingSpots = () => {
  const [spots, setSpots] = useState([]);

  useEffect(() => {
    // This will create the collection if it doesn't exist
    const spotsRef = collection(db, 'parking-lots', 'main-lot', 'spots');
    
    const unsubscribe = onSnapshot(spotsRef, (snapshot) => {
      const spotsData = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
      setSpots(spotsData);
    });

    return unsubscribe;
  }, []);

  // Initialize spots if empty (run once)
  const initializeSpots = async (totalSpots = 20) => {
    for (let i = 1; i <= totalSpots; i++) {
      const spotRef = doc(db, 'parking-lots', 'main-lot', 'spots', `spot-${i}`);
      await setDoc(spotRef, {
        number: i,
        occupied: Math.random() > 0.5, // random initial state
        confidence: 0.9,
        lastUpdated: new Date(),
        section: i <= 10 ? 'A' : 'B'
      });
    }
  };

  return { spots, initializeSpots };
};