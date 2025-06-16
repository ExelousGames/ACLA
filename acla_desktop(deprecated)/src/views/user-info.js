import React, { useEffect, useState } from 'react';
import axios from 'axios';

const UserInfo = () => {
    const [tasks, setTasks] = useState([]);

    const serverIPandPort = process.env.REACT_APP_BACKEND_SERVER_IP + ":" + process.env.REACT_APP_BACKEND_PROXY_PORT
    const server_url_header = 'http://' + serverIPandPort


    useEffect(() => {
        axios.get(server_url_header + '/userinfo/${id}')
            .then(response => {
                setTasks(response.data)
            })
            .catch(error => console.error('Error fetching tasks:', error));
    }, []);

    const deleteTask = (id) => {
        axios.delete(server_url_header + +`/userinfo/${id}`)
            .then(() => setTasks(tasks.filter(task => task.id !== id)))
            .catch(error => console.error('Error deleting task:', error));
    };

    return (
        <div>

        </div>
    );
};

export default UserInfo;