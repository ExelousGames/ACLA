import React, { useState } from 'react';
import axios from 'axios';

const CreateUserInfo = ({ onTaskCreated }) => {
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        axios.post('http://localhost:7001/userinfo', { infoDto: { name: name } })
            .then(response => {
                setName('');
                setDescription('');
            })
            .catch(error => console.error('Error creating task:', error));
    };

    return (
        <form onSubmit={handleSubmit}>
            <input
                type="text"
                placeholder="Name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
            />
            <textarea
                placeholder="Description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
            ></textarea>
            <button type="submit">Add Task</button>
        </form>
    );
};

export default CreateUserInfo;