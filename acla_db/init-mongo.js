db.createCollection("userinfos");

db.createUser({
    user: 'client',
    pwd: 'clientpass',
    roles: [
        {
            role: 'readWrite',
            db: 'ACLA',
        },
    ],
});