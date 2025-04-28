import React, { useEffect, useState } from 'react';
import axios from 'axios';

const UserInfo = () => {
    const [tasks, setTasks] = useState([]);

    useEffect(() => {
        axios.get('http://localhost:7001/userinfo/${id}')
            .then(response => {
                setTasks(response.data)
            })
            .catch(error => console.error('Error fetching tasks:', error));
    }, []);

    const deleteTask = (id) => {
        axios.delete(`http://localhost:7001/userinfo/${id}`)
            .then(() => setTasks(tasks.filter(task => task.id !== id)))
            .catch(error => console.error('Error deleting task:', error));
    };

    return (
        <div>
            <h1>Task List</h1>
        </div>
    );
};

export default UserInfo;