import React, { FC } from 'react';


type PageHeaderProps = {
    title: string;
    subtitle?: string;
};

const PageHeader: FC<PageHeaderProps> = ({ title, subtitle }) => {
    return (
        <div className="ap-page-heading">
            <h1 className="ap-page-title">
                {title}
            </h1>
            {subtitle && (
                <h2 className="ap-page-desc">
                    {subtitle}
                </h2>
            )}
        </div>
    );
};

export default PageHeader;