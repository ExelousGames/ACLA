import logo from './logo.svg';
import './App.css';
import React, { useState } from 'react';
import UserInfo from './user-info';
import CreateUserInfo from './create-user-info';



function App() {
  const [tasks, setTasks] = useState([]);

  const handleTaskCreated = (newTask) => {
    setTasks([...tasks, newTask]);
  };

  return (
    <div className="App">
      <UserInfo onTaskCreated={handleTaskCreated} />
      <CreateUserInfo />
    </div>
  );
}

export default App;
