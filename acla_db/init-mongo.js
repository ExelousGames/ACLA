db.createCollection("userinfos");
db.createCollection("racingmaps");
db.createCollection("racingsessions");

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