db.createUser({
    user: 'ddd',
    pwd: 'ddd',
    roles: [
        {
            role: 'readWrite',
            db: 'ACLA',
        },
    ],
});