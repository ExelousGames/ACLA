import './user-profile.css';

import {
    Avatar,
    Box,
    Button,
    Card,
    Flex,
    Heading,
    Text,
    TextField,
    Separator,
    Strong,
    Badge,
    Container
} from "@radix-ui/themes";
import { useAuth } from 'hooks/AuthProvider';
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const UserProfile = () => {
    const auth = useAuth();
    const navigate = useNavigate();
    const [isEditing, setIsEditing] = useState(false);
    const [formData, setFormData] = useState({
        firstName: '',
        lastName: '',
        email: '',
        username: ''
    });

    useEffect(() => {
        if (auth?.userProfile) {
            setFormData({
                firstName: auth.userProfile.firstName || '',
                lastName: auth.userProfile.lastName || '',
                email: auth.userProfile.email || auth.user || '',
                username: auth.userProfile.username || ''
            });
        }
    }, [auth?.userProfile, auth?.user]);

    const handleInputChange = (field: string, value: string) => {
        setFormData(prev => ({
            ...prev,
            [field]: value
        }));
    };

    const handleSave = () => {
        // TODO: Implement save functionality with API call
        setIsEditing(false);
        // You can add API call here to save the profile data
    };

    const handleCancel = () => {
        // Reset form data to original values
        if (auth?.userProfile) {
            setFormData({
                firstName: auth.userProfile.firstName || '',
                lastName: auth.userProfile.lastName || '',
                email: auth.userProfile.email || auth.user || '',
                username: auth.userProfile.username || ''
            });
        }
        setIsEditing(false);
    };

    const getInitials = () => {
        const firstName = formData.firstName || '';
        const lastName = formData.lastName || '';
        return `${firstName.charAt(0)}${lastName.charAt(0)}`.toUpperCase() || 'U';
    };

    const getUserRoles = () => {
        if (auth?.userProfile?.roles && Array.isArray(auth.userProfile.roles)) {
            return auth.userProfile.roles.map((role: any) => role.name || role).join(', ');
        }
        return 'No roles assigned';
    };

    const handleBackToDashboard = () => {
        navigate('/dashboard');
    };

    return (
        <Container className="user-profile-container" maxWidth="600px">
            <Card className="user-profile-card">
                <Flex direction="column" gap="4">
                    {/* Back Button */}
                    <Flex justify="start">
                        <Button variant="soft" onClick={handleBackToDashboard}>
                            ← Back to Dashboard
                        </Button>
                    </Flex>

                    {/* Header */}
                    <Flex justify="between" align="center">
                        <Heading size="6">User Profile</Heading>
                        {!isEditing ? (
                            <Button onClick={() => setIsEditing(true)}>
                                Edit Profile
                            </Button>
                        ) : (
                            <Flex gap="2">
                                <Button variant="soft" onClick={handleCancel}>
                                    Cancel
                                </Button>
                                <Button onClick={handleSave}>
                                    Save
                                </Button>
                            </Flex>
                        )}
                    </Flex>

                    <Separator />

                    {/* Profile Picture and Basic Info */}
                    <Flex direction="column" align="center" gap="3">
                        <Avatar size="9" fallback={getInitials()} />
                        <Box>
                            <Text size="5" weight="bold">
                                {formData.firstName && formData.lastName
                                    ? `${formData.firstName} ${formData.lastName}`
                                    : formData.username || formData.email || 'User'
                                }
                            </Text>
                        </Box>
                        <Badge variant="soft" color="blue">
                            {getUserRoles()}
                        </Badge>
                    </Flex>

                    <Separator />

                    {/* Profile Form */}
                    <Flex direction="column" gap="4">
                        <Heading size="4">Personal Information</Heading>

                        <Flex direction="column" gap="3">
                            <Box>
                                <Text size="2" color="gray" mb="1">
                                    <Strong>First Name</Strong>
                                </Text>
                                {isEditing ? (
                                    <TextField.Root
                                        value={formData.firstName}
                                        onChange={(e) => handleInputChange('firstName', e.target.value)}
                                        placeholder="Enter your first name"
                                    />
                                ) : (
                                    <Text size="3">{formData.firstName || 'Not provided'}</Text>
                                )}
                            </Box>

                            <Box>
                                <Text size="2" color="gray" mb="1">
                                    <Strong>Last Name</Strong>
                                </Text>
                                {isEditing ? (
                                    <TextField.Root
                                        value={formData.lastName}
                                        onChange={(e) => handleInputChange('lastName', e.target.value)}
                                        placeholder="Enter your last name"
                                    />
                                ) : (
                                    <Text size="3">{formData.lastName || 'Not provided'}</Text>
                                )}
                            </Box>

                            <Box>
                                <Text size="2" color="gray" mb="1">
                                    <Strong>Username</Strong>
                                </Text>
                                {isEditing ? (
                                    <TextField.Root
                                        value={formData.username}
                                        onChange={(e) => handleInputChange('username', e.target.value)}
                                        placeholder="Enter your username"
                                    />
                                ) : (
                                    <Text size="3">{formData.username || 'Not provided'}</Text>
                                )}
                            </Box>

                            <Box>
                                <Text size="2" color="gray" mb="1">
                                    <Strong>Email</Strong>
                                </Text>
                                {isEditing ? (
                                    <TextField.Root
                                        type="email"
                                        value={formData.email}
                                        onChange={(e) => handleInputChange('email', e.target.value)}
                                        placeholder="Enter your email"
                                    />
                                ) : (
                                    <Text size="3">{formData.email || 'Not provided'}</Text>
                                )}
                            </Box>
                        </Flex>

                        <Separator />

                        {/* Account Information */}
                        <Heading size="4">Account Information</Heading>

                        <Box>
                            <Text size="2" color="gray" mb="1">
                                <Strong>Roles</Strong>
                            </Text>
                            <Text size="3">{getUserRoles()}</Text>
                        </Box>

                        {auth?.userProfile?.permissions && (
                            <Box>
                                <Text size="2" color="gray" mb="1">
                                    <Strong>Permissions</Strong>
                                </Text>
                                <Text size="3">
                                    {Array.isArray(auth.userProfile.permissions)
                                        ? auth.userProfile.permissions.length + ' permissions assigned'
                                        : 'No permissions assigned'
                                    }
                                </Text>
                            </Box>
                        )}
                    </Flex>
                </Flex>
            </Card>
        </Container>
    );
};

export default UserProfile;
